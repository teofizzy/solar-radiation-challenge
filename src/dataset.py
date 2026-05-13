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
import torch
from torch.utils.data import Dataset

from src.config import HPARAMS, DTYPE, PATHS


# Feature columns used as model input (ORDER MATTERS -- must be consistent)
ASTRO_FEATURES = ['cos_zenith', 'solar_zenith', 'clear_sky_ghi', 'log_clearsky_ghi']

# ERA5: Use Celsius conversions instead of raw Kelvin t2m/d2m.
# Raw sp stays (z-range ~4.7, well-behaved). Raw tco3/tcwv stay.
ERA5_FEATURES = ['u10', 'v10', 'd2m_celsius', 't2m_celsius', 'sp', 'tco3', 'tcwv']

PHYSICS_FEATURES = [
    'air_mass', 'wind_speed', 'log_wind_speed',
    'wind_direction_sin', 'wind_direction_cos',
    'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
    'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos', 
    'hour_6_sin', 'hour_6_cos', 'hour_3_sin', 'hour_3_cos',
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'days_since_start'
]

# Use log_precipitation instead of raw precipitation (z-range 62 -> ~5)
LOCAL_FEATURES = ['temperature', 'relativehumidity', 'log_precipitation']

# Temporal features (rolling stats) -- added dynamically if present
# Not listed here since they are generated programmatically

LANDSAF_FEATURES = ['mdssf', 'mlst', 'kt_landsaf']
TROPOMI_FEATURES = ['tropomi_cloud', 'tropomi_cloud_missing', 'tropomi_cloud_age_hours', 
                    'tropomi_aerosol', 'tropomi_aerosol_missing', 'tropomi_aerosol_age_hours']

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Determine which feature columns are available in the DataFrame.
    Returns the ordered list of feature column names for model input.
    """
    candidates = ASTRO_FEATURES + ERA5_FEATURES + PHYSICS_FEATURES + LOCAL_FEATURES + LANDSAF_FEATURES + TROPOMI_FEATURES

    # Add any temporal rolling columns
    rolling_cols = [c for c in df.columns if '_roll_' in c or '_diff_' in c or
                    c in ('volatility_index', 'hours_since_wash',
                          'sticky_dust_index')]
    candidates += sorted(rolling_cols)

    # Add satellite lag stacks (e.g., mdssf_lag_1, kt_landsaf_lag_4)
    lag_cols = [c for c in df.columns if '_lag_' in c]
    candidates += sorted(lag_cols)

    # Add EWMA drift features
    ewma_cols = [c for c in df.columns if c.startswith('ewma_kt_') or
                 c in ('drift_proxy', 'log_cum_exposure')]
    candidates += sorted(ewma_cols)

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
        target_kt: float -- clearness index at center (NaN for test)
        target_ghi: float -- radiation at center (NaN for test)

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
                 is_train: bool = True, scaler_stats: dict = None):
        self.half_window = HPARAMS['half_window']
        self.seq_len = HPARAMS['seq_len']
        self.feature_cols = feature_cols
        self.is_train = is_train

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

            # Target: kt (NaN for test rows)
            if 'kt' in st_df.columns:
                target_kt = st_df['kt'].values.astype(np.float32)
            else:
                target_kt = np.full(n_rows, np.nan, dtype=np.float32)

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
                self.samples.append({
                    'feat_matrix': feat_matrix,
                    'center': center,
                    'station_idx': station_idx,
                    'clear_sky_ghi': clear_sky[center],
                    'is_night': is_night[center],
                    'target_kt': target_kt[center],
                    'target_ghi': target_ghi[center],
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
        """Compute normalization stats from TRAINING data only."""
        if self.is_train:
            # Use only odd months (training months) for scaler
            train_mask = df['month'].isin([1, 3, 5, 7, 9, 11])
            train_data = df.loc[train_mask, feature_cols]
        else:
            train_data = df[feature_cols]

        self.mean = train_data.mean().values.astype(np.float32)
        self.std = train_data.std().values.astype(np.float32)
        # Prevent division by zero; use 1.0 for near-constant features
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        # Robust scaling override for heavy-tailed features.
        # If any feature's z-range > 15 (indicating extreme outliers),
        # switch from mean/std to median/IQR for that feature.
        q25 = train_data.quantile(0.25).values.astype(np.float32)
        q75 = train_data.quantile(0.75).values.astype(np.float32)
        iqr = q75 - q25
        median = train_data.median().values.astype(np.float32)

        for i in range(len(feature_cols)):
            if self.std[i] > 1e-6:
                z_range = (train_data.iloc[:, i].max() - train_data.iloc[:, i].min()) / self.std[i]
                if z_range > 15 and iqr[i] > 1e-6:
                    # Switch to robust: center on median, scale by IQR
                    self.mean[i] = median[i]
                    self.std[i] = iqr[i]

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

        return {
            'x': torch.from_numpy(window),                              # (seq_len, n_features)
            'station_idx': torch.tensor(sample['station_idx'], dtype=torch.long),
            'clear_sky_ghi': torch.tensor(sample['clear_sky_ghi'], dtype=torch.float32),
            'is_night': torch.tensor(sample['is_night'], dtype=torch.float32),
            'target_kt': torch.tensor(sample['target_kt'], dtype=torch.float32),
            'target_ghi': torch.tensor(sample['target_ghi'], dtype=torch.float32),
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
        if sample['is_test'] == 1 or np.isnan(sample['target_kt']):
            continue
        # Check if this is a val sample
        sample_id = sample['sample_id']
        # Use month from the original data
        # We need another way to check -- use the fact that center timestep
        # maps back to the original df
        # For simplicity, store month in samples during dataset construction
        # Actually, let's filter by checking target availability and test flag
        if not np.isnan(sample['target_kt']):
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
                        scaler_stats: dict):
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
                        scaler_stats=scaler_stats)
