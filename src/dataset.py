"""
PyTorch Dataset for the Physics-Informed BiLSTM.
Symmetric sliding windows for retrospective reconstruction.

Key design decisions:
- SYMMETRIC windows: 24 past + 24 future = 48 steps (12hr at 15-min cadence)
- Target is clearness index kt at the CENTER timestep
- Normalization stats computed on TRAINING months only
- No target radiation in the input feature tensor
"""

import os
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset: pass # Dummy for type checking

from src.config import HPARAMS, DTYPE, PATHS, DIAGNOSTIC_FEATURES


# Feature columns used as model input (ORDER MATTERS -- must be consistent)
ASTRO_FEATURES = ['cos_zenith', 'csghi_terrain_corr']

# ERA5: Use terrain-corrected versions (lapse rate, hypsometry) + missing flag
ERA5_FEATURES = ['u10', 'v10', 't_lapse_corr', 'p_hyps_corr', 'tco3', 'tcwv', 'era5_missing']

PHYSICS_FEATURES = [
    'wind_direction_sin', 'wind_direction_cos',
    'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
    'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos', 
    'hour_6_sin', 'hour_6_cos',
    'doy_sin', 'doy_cos', 'days_since_start'
]

# Use log_precipitation instead of raw precipitation (z-range 62 -> ~5)
LOCAL_FEATURES = ['temperature', 'relativehumidity', 'log_precipitation']

# Temporal features (rolling stats) -- added dynamically if present
# Not listed here since they are generated programmatically

LANDSAF_FEATURES = ['mdssf', 'mlst', 'kt_landsaf']
TROPOMI_FEATURES = ['tropomi_cloud', 'tropomi_cloud_missing', 'tropomi_cloud_age_hours', 
                    'tropomi_aerosol', 'tropomi_aerosol_missing', 'tropomi_aerosol_age_hours']

# Static: Keep only 'dist_water' (lake/sea breeze) as raw feature.
# Others (dem, slope, aspect) are now baked into physics features.
STATIC_FEATURES = ['dist_water']
# Land use OHE columns will be added dynamically by the 'lu_' prefix check

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Determine which feature columns are available in the DataFrame.
    Returns the ordered list of feature column names for model input.
    """
    candidates = (ASTRO_FEATURES + ERA5_FEATURES + PHYSICS_FEATURES + 
                  LOCAL_FEATURES + LANDSAF_FEATURES + TROPOMI_FEATURES + STATIC_FEATURES)

    # Add Land Use OHE (e.g. lu_12)
    lu_cols = [c for c in df.columns if c.startswith('lu_')]
    candidates += sorted(lu_cols)

    # Add any temporal rolling columns
    rolling_cols = [c for c in df.columns if '_roll_' in c or '_diff_' in c or
                    c in ('volatility_index', 'hours_since_wash',
                          'sticky_dust_index', 'advection_kt')]
    candidates += sorted(rolling_cols)

    # Add satellite lag stacks (e.g., mdssf_lag_1, kt_landsaf_lag_4)
    lag_cols = [c for c in df.columns if '_lag_' in c]
    candidates += sorted(lag_cols)

    # Add EWMA drift features (exclude raw fast/slow components as redundant with drift_proxy)
    ewma_cols = [c for c in df.columns if (c.startswith('ewma_kt_') and c not in ['ewma_kt_fast', 'ewma_kt_slow']) or
                 c in ('drift_proxy', 'log_cum_exposure', 'clearness_regime_shift')]
    candidates += sorted(ewma_cols)

    # Add Interaction features
    inter_cols = ['zenith_humidity', 'zenith_cloud', 'airmass_aerosol', 'airmass_water']
    candidates += [c for c in inter_cols if c in df.columns]

    # Filter to only available columns
    available = [c for c in candidates if c in df.columns]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in available:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


class SolarDataset(Dataset):
    """
    PyTorch Dataset producing symmetric sliding windows for BiLSTM reconstruction.

    Each sample:
        x: (seq_len, n_features) -- covariate window
        station_idx: int -- station index for embedding
        clear_sky_ghi: float -- clear-sky GHI at center timestep
        is_night: int -- nighttime flag at center timestep
        target_delta_kt: float -- clearness index residual at center (NaN for test)
        target_ghi: float -- radiation at center (NaN for test)
        center_kt_landsaf: float -- LandSAF clearness index at center (available everywhere)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (single station or all stations).
    feature_cols : list
        Ordered list of feature column names.
    is_train : bool
        If True, return targets. If False, return NaN targets.
    scaler_stats : dict or None
        Pre-computed {'mean': array, 'std': array} for normalization.
        If None and is_train=True, will compute from this data.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 is_train: bool = True, scaler_stats: dict = None, hparams: dict = None):
        
        # Use provided hparams or fallback to global HPARAMS
        self.hparams = hparams if hparams is not None else HPARAMS
        self.seq_len = self.hparams['seq_len']
        self.half_window = self.seq_len // 2
        self.feature_cols = feature_cols
        self.is_train = is_train

        # Load Diagnostic Descriptors
        diag_path = os.path.join(PATHS['cache_raw'], 'station_diagnostic_summary.csv')
        if os.path.exists(diag_path):
            diag_df = pd.read_csv(diag_path)
            # Normalize diagnostic features internally (Min-Max)
            for col in DIAGNOSTIC_FEATURES:
                if col in diag_df.columns:
                    c_min, c_max = diag_df[col].min(), diag_df[col].max()
                    diag_df[col] = (diag_df[col] - c_min) / (c_max - c_min + 1e-9)
            self.diag_map = diag_df.set_index('station')[DIAGNOSTIC_FEATURES].to_dict('index')
        else:
            print(f"WARNING: {diag_path} not found. Using zero diagnostic vectors.")
            self.diag_map = {}

        # Sort by station and timestamp to ensure temporal ordering
        df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)

        # Build per-station arrays for efficient windowing
        self.samples = []
        self.station_boundaries = {}

        stations = sorted(df['station'].unique())

        for station_id in stations:
            st_mask = df['station'] == station_id
            st_df = df.loc[st_mask]

            station_idx = st_df['station_idx'].iloc[0]
            n_rows = len(st_df)

            # Feature matrix for this station
            feat_matrix = st_df[feature_cols].values.astype(np.float32)

            # Metadata at each timestep
            clear_sky = st_df['clear_sky_ghi'].values.astype(np.float32)
            is_night = st_df['is_night'].values.astype(np.int8)

            # Target: Delta kt = kt_obs - kt_landsaf (NaN for test rows)
            if 'kt' in st_df.columns and 'kt_landsaf' in st_df.columns:
                target_delta_kt = (st_df['kt'].values - st_df['kt_landsaf'].values).astype(np.float32)
            else:
                target_delta_kt = np.full(n_rows, np.nan, dtype=np.float32)
                
            # Need kt_landsaf at center to reconstruct kt from delta_kt
            if 'kt_landsaf' in st_df.columns:
                center_kt_landsaf = st_df['kt_landsaf'].values.astype(np.float32)
            else:
                center_kt_landsaf = np.full(n_rows, np.nan, dtype=np.float32)

            # Raw radiation target
            if 'radiation' in st_df.columns:
                target_ghi = st_df['radiation'].values.astype(np.float32)
            else:
                target_ghi = np.full(n_rows, np.nan, dtype=np.float32)

            # IDs for submission
            if 'ID' in st_df.columns:
                ids = st_df['ID'].values
            else:
                ids = np.array([''] * n_rows)

            # Generate valid window centers
            for center in range(self.half_window, n_rows - self.half_window):
                # Get diagnostic vector
                diag_vec = self.diag_map.get(station_id, {c: 0.0 for c in DIAGNOSTIC_FEATURES})
                diag_tensor = np.array([diag_vec[c] for c in DIAGNOSTIC_FEATURES], dtype=np.float32)

                self.samples.append({
                    'feat_matrix': feat_matrix,
                    'is_night_full': is_night,
                    'center': center,
                    'station_idx': station_idx,
                    'diag_vector': diag_tensor,
                    'clear_sky_ghi': clear_sky[center],
                    'is_night': is_night[center],
                    'target_delta_kt': target_delta_kt[center],
                    'target_ghi': target_ghi[center],
                    'center_kt_landsaf': center_kt_landsaf[center],
                    'sample_id': ids[center],
                    'is_test': st_df['is_test'].iloc[center],
                    'year': st_df['year'].iloc[center],
                    'month': st_df['month'].iloc[center],
                })

        # Compute or load normalization statistics
        if scaler_stats is not None:
            self.mean = scaler_stats['mean']
            self.std = scaler_stats['std']
        else:
            self._compute_scaler(df, feature_cols)

        print(f"  SolarDataset: {len(self.samples)} samples, "
              f"{len(feature_cols)} features, "
              f"window={self.seq_len}")

    def _compute_scaler(self, df: pd.DataFrame, feature_cols: list):
        """
        Compute robust normalization stats (Median/IQR) from TRAINING data only.
        Weather data is heavy-tailed; mean/std is too sensitive to outliers.
        """
        if self.is_train:
            # Use only odd months (training months) for scaler
            train_mask = df['month'].isin([1, 3, 5, 7, 9, 11])
            train_data = df.loc[train_mask, feature_cols]
        else:
            train_data = df[feature_cols]

        # Robust Scaling: center on median, scale by IQR
        self.mean = train_data.median().to_numpy().astype(np.float32)
        q25 = train_data.quantile(0.25).to_numpy().astype(np.float32)
        q75 = train_data.quantile(0.75).to_numpy().astype(np.float32)
        self.std = (q75 - q25).astype(np.float32)
        
        # Prevent division by zero; use 1.0 for constant or binary features
        # Weather data can have zero variance in small windows/batches
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        print(f"  [SCALER] Global Robust Scaling (Median/IQR) active for {len(feature_cols)} features.")

    def get_scaler_stats(self) -> dict:
        """Return scaler statistics for reuse in test dataset."""
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center = sample['center']
        hw = self.half_window

        # Extract symmetric window
        window = sample['feat_matrix'][center - hw:center + hw]  # (seq_len, n_features)

        # Z-score normalize
        window = (window - self.mean) / self.std

        # Handle any NaN in features (replace with 0 after normalization)
        window = np.nan_to_num(window, nan=0.0)

        x_tensor = torch.from_numpy(window)
        is_night_seq = sample['is_night_full'][center - hw:center + hw] # (seq_len,)
        is_night_tensor = torch.from_numpy(is_night_seq).float()
        
        # 8. Augmentations (Training Only)
        if self.is_train:
            from src.augment import apply_intensity_jitter, apply_temporal_mask
            x_tensor = apply_intensity_jitter(x_tensor, is_night_tensor, p=0.4)
            x_tensor = apply_temporal_mask(x_tensor, p=0.2)

        return {
            'x': x_tensor,                              # (seq_len, n_features)
            'station_idx': torch.tensor(sample['station_idx'], dtype=torch.long),
            'diag_vector': torch.from_numpy(sample['diag_vector']),     # (5,)
            'clear_sky_ghi': torch.tensor(sample['clear_sky_ghi'], dtype=torch.float32),
            'is_night': torch.tensor(sample['is_night'], dtype=torch.float32),
            'target_delta_kt': torch.tensor(sample['target_delta_kt'], dtype=torch.float32),
            'target_ghi': torch.tensor(sample['target_ghi'], dtype=torch.float32),
            'center_kt_landsaf': torch.tensor(sample['center_kt_landsaf'], dtype=torch.float32),
            'is_test': torch.tensor(sample['is_test'], dtype=torch.uint8),
            'sample_id': sample['sample_id'],
        }


def create_train_val_datasets(df: pd.DataFrame, feature_cols: list,
                              val_year: int = 2017):
    """
    Create train and validation datasets using temporal CV.

    Train: odd months (1,3,5,7,9,11) of val_year + all non-val_year data
    Val: even months (2,4,6,8,10,12) of val_year

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    feature_cols : list
        Feature column names.
    val_year : int
        Year to use for validation split.

    Returns
    -------
    train_dataset, val_dataset : SolarDataset
    """
    # Training data: rows with valid target (not test) AND not in val set
    has_target = df['radiation'].notna()

    # Validation: months specified in val_months
    # Note: `val_months` logic is handled during dataset split in train.py, 
    # but for initial dataset creation, we just create the full temporal dataset.
    
    # We actually don't need a val_mask here if we rely on `get_train_val_indices` 
    # in train.py to do the splitting. We just pass the dataset.
    df_train = df.copy()
    df_val = df.copy()

    # But we need the full temporal context (including test rows) for windowing
    # So we pass full station data but mark which rows are trainable
    df_train = df.copy()
    df_val = df.copy()

    # For training dataset: only include samples where center has valid target
    # and center is in a training month
    print(f"\n[DATASET] Creating temporal CV split (val_year={val_year})...")
    print(f"  Train targets: {train_mask.sum():,}")
    print(f"  Val targets:   {val_mask.sum():,}")

    # Create train dataset (all data for windowing, but only train targets)
    train_dataset = SolarDataset(
        df_train, feature_cols, is_train=True, scaler_stats=None
    )

    # Filter train dataset samples to only training targets
    train_indices = []
    val_indices = []
    for i, sample in enumerate(train_dataset.samples):
        if sample['is_test'] == 1 or np.isnan(sample['target_delta_kt']):
            continue
        # Check if this is a val sample
        sample_id = sample['sample_id']
        # Use month from the original data
        # We need another way to check -- use the fact that center timestep
        # maps back to the original df
        # For simplicity, store month in samples during dataset construction
        # Actually, let's filter by checking target availability and test flag
        if not np.isnan(sample['target_delta_kt']):
            # We'll split by index later; for now collect all valid samples
            train_indices.append(i)

    # Get scaler from train
    scaler_stats = train_dataset.get_scaler_stats()

    # Create val dataset with same scaler
    val_dataset = SolarDataset(
        df_val, feature_cols, is_train=False, scaler_stats=scaler_stats
    )

    return train_dataset, val_dataset, scaler_stats


def create_test_dataset(df: pd.DataFrame, feature_cols: list,
                        scaler_stats: dict, hparams: dict = None):
    """
    Create test dataset for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (full data for context).
    feature_cols : list
        Feature column names.
    scaler_stats : dict
        Normalization stats from training.

    Returns
    -------
    SolarDataset
    """
    return SolarDataset(df, feature_cols, is_train=False,
                        scaler_stats=scaler_stats, hparams=hparams)
