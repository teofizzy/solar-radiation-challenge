"""
PyTorch Dataset for the Physics-Informed BiLSTM.
Symmetric sliding windows for retrospective reconstruction.

Key design decisions (V1 reverted):
- SYMMETRIC windows: 24 past + 24 future = 48 steps (12hr at 15-min cadence)
- Target is clearness index kt at the CENTER timestep (NOT delta_kt)
- Normalization: Median/IQR (proven stable in V1)
- No augmentation (Multi-AI consensus: risky in solar physics)
- No atmos_feats, no diag_vector (removed over-engineering)
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
    class Dataset: pass

from src.config import HPARAMS, DTYPE, PATHS


# Feature columns used as model input (ORDER MATTERS -- must be consistent)
ASTRO_FEATURES = ['cos_zenith', 'csghi_terrain_corr']

# ERA5: Use terrain-corrected versions (lapse rate, hypsometry) + missing flag
ERA5_FEATURES = ['u10', 'v10', 't_lapse_corr', 'p_hyps_corr', 'tco3', 'tcwv', 'era5_missing']

# Physics-derived features
PHYSICS_FEATURES = [
    'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
    'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos', 
    'hour_6_sin', 'hour_6_cos',
    'doy_sin', 'doy_cos', 'days_since_start'
]

# Local station measurements
LOCAL_FEATURES = ['temperature', 'relativehumidity', 'log_precipitation']

# Temporal features (rolling stats) -- added dynamically if present
# Not listed here since they are generated programmatically

LANDSAF_FEATURES = ['mdssf', 'mlst', 'kt_landsaf']
TROPOMI_FEATURES = ['tropomi_cloud', 'tropomi_cloud_missing', 'tropomi_cloud_age_hours', 
                    'tropomi_aerosol', 'tropomi_aerosol_missing', 'tropomi_aerosol_age_hours']

# Static: Keep only 'dist_water' (lake/sea breeze) as raw feature.
STATIC_FEATURES = ['dist_water']


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

    # Add EWMA drift features
    ewma_cols = [c for c in df.columns if (c.startswith('ewma_kt_') and c not in ['ewma_kt_fast', 'ewma_kt_slow']) or
                 c in ('drift_proxy', 'log_cum_exposure', 'clearness_regime_shift')]
    candidates += sorted(ewma_cols)

    # Add Interaction features (pruned: only airmass_aerosol kept per consensus)
    inter_cols = ['airmass_aerosol']
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
        is_night: float -- nighttime flag at center timestep
        target_kt: float -- clearness index at center (NaN for test)
        target_ghi: float -- radiation at center (NaN for test)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (single station or all stations).
    feature_cols : list
        Ordered list of feature column names.
    is_train : bool
        If True, compute normalization stats from training months.
    scaler_stats : dict or None
        Pre-computed {'center': array, 'scale': array} for normalization.
    """

    # ---- Feature categories for hybrid normalization ----
    # Scientific backing: PISSM paper, ERA5_Ag paper, Perplexity deep search,
    # ChatGPT consensus, NotebookLM sources [5-13].
    # Different feature families have fundamentally different distributions.
    
    _NO_SCALE_PREFIXES = {
        # Physics-bounded [-1,1] or [0,1.05]: already in optimal range
        'cos_zenith', 'kt_landsaf',
        # Temporal cycles: deterministic, already [-1,1]
        'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos',
        'hour_6_sin', 'hour_6_cos', 'doy_sin', 'doy_cos',
        # Binary flags: 0/1
        'era5_missing', 'tropomi_cloud_missing', 'tropomi_aerosol_missing',
        'is_night',
        # Static per-station: constant within a station's window
        'lu_',
    }
    _MINMAX_FEATURES = {
        # Bounded positive: clear-sky GHI [0, ~1200], days_since_start [0, N]
        'csghi_terrain_corr', 'days_since_start',
        # Satellite irradiance products (bounded positive)
        'mdssf', 'mlst',
        # Static distance (bounded positive)
        'dist_water',
    }
    _ROBUST_FEATURES = {
        # Heavy-tailed: exact column names only (no prefix matching)
        'log_precipitation', 'tropomi_aerosol',
    }
    # Everything else: Z-score (ERA5, local weather, rolling stats, lags, EWMA, etc.)

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 is_train: bool = True, scaler_stats: dict = None, hparams: dict = None):
        
        # Use provided hparams or fallback to global HPARAMS
        self.hparams = hparams if hparams is not None else HPARAMS
        self.seq_len = self.hparams['seq_len']
        self.half_window = self.seq_len // 2
        self.feature_cols = feature_cols
        self.is_train = is_train

        # Sort by station and timestamp to ensure temporal ordering
        df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)

        # Build per-station arrays for efficient windowing
        self.samples = []

        stations = sorted(df['station'].unique())

        for station_id in stations:
            st_mask = df['station'] == station_id
            st_df = df.loc[st_mask]

            station_idx = st_df['station_idx'].iloc[0]
            n_rows = len(st_df)

            # Feature matrix for this station
            feat_matrix = st_df[feature_cols].values.astype(np.float32)

            # Metadata at each timestep
            clear_sky = st_df['clear_sky_ghi'].values.astype(np.float32) if 'clear_sky_ghi' in st_df.columns else np.zeros(n_rows, dtype=np.float32)
            is_night = st_df['is_night'].values.astype(np.float32) if 'is_night' in st_df.columns else np.zeros(n_rows, dtype=np.float32)

            # Target: kt (clearness index), NOT delta_kt
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
            self.center = scaler_stats['center']
            self.scale = scaler_stats['scale']
        else:
            self._compute_scaler(df, feature_cols)

        print(f"  SolarDataset: {len(self.samples)} samples, "
              f"{len(feature_cols)} features, "
              f"window={self.seq_len}")

    @classmethod
    def _get_feature_category(cls, col: str) -> str:
        """Classify a feature column into its normalization category.
        
        Categories (multi-AI consensus + literature):
            'none'   - Physics-bounded, temporal cycles, binary flags
            'minmax' - Bounded positive (clear-sky, satellite, distances)
            'robust' - Heavy-tailed distributions (aerosol, precipitation)
            'zscore' - Everything else (ERA5, local weather, rolling stats)
        """
        # Priority 1: No-scale (exact match or prefix match for lu_)
        for prefix in cls._NO_SCALE_PREFIXES:
            if col == prefix or col.startswith(prefix):
                return 'none'
        # Priority 2: Min-Max (exact match or startswith)
        if col in cls._MINMAX_FEATURES or any(col.startswith(p) for p in cls._MINMAX_FEATURES):
            return 'minmax'
        # Priority 3: Robust (exact match ONLY -- prevents tropomi_aerosol matching tropomi_aerosol_age_hours)
        if col in cls._ROBUST_FEATURES:
            return 'robust'
        # Default: Z-score
        return 'zscore'

    def _compute_scaler(self, df: pd.DataFrame, feature_cols: list):
        """
        Physics-aware hybrid normalization (multi-AI consensus).
        
        Different feature families get different treatment:
          - No scaling: physics-bounded, temporal cycles, binary flags
          - Min-Max: clear-sky GHI, satellite products, static distances
          - Robust (Median/IQR): heavy-tailed aerosol/precipitation
          - Z-score (mean/std): ERA5, local weather, rolling stats, lags
          
        All statistics computed on TRAINING months only (odd months).
        Global normalization (ERA5_Ag paper: identical to per-station).
        
        References:
          - PISSM paper: Z-score for meteorological, no-scale for SZA
          - ERA5_Ag paper: global norm preferred, same performance
          - Perplexity: "Min-Max for kt, Robust for aerosol, Z-score for ERA5"
        """
        if self.is_train:
            train_mask = df['month'].isin([1, 3, 5, 7, 9, 11])
            train_data = df.loc[train_mask, feature_cols]
        else:
            train_data = df[feature_cols]

        n_feats = len(feature_cols)
        self.center = np.zeros(n_feats, dtype=np.float32)
        self.scale = np.ones(n_feats, dtype=np.float32)

        counts = {'none': 0, 'zscore': 0, 'minmax': 0, 'robust': 0}

        for i, col in enumerate(feature_cols):
            cat = self._get_feature_category(col)
            counts[cat] += 1
            col_data = train_data[col].dropna()

            if cat == 'none':
                # No scaling: center=0, scale=1 (identity transform)
                self.center[i] = 0.0
                self.scale[i] = 1.0

            elif cat == 'minmax':
                # Min-Max: center=min, scale=max-min -> maps to [0, 1]
                c_min = col_data.min()
                c_max = col_data.max()
                self.center[i] = np.float32(c_min)
                self.scale[i] = np.float32(max(c_max - c_min, 1e-6))

            elif cat == 'robust':
                # Robust: center=median, scale=IQR (insensitive to outliers)
                self.center[i] = np.float32(col_data.median())
                q25 = np.float32(col_data.quantile(0.25))
                q75 = np.float32(col_data.quantile(0.75))
                iqr = q75 - q25
                self.scale[i] = np.float32(max(iqr, 1e-6))

            else:  # zscore
                # Z-score: center=mean, scale=std
                self.center[i] = np.float32(col_data.mean())
                col_std = np.float32(col_data.std())
                self.scale[i] = np.float32(max(col_std, 1e-6))

        print(f"  [SCALER] Hybrid normalization for {n_feats} features:")
        print(f"    No-scale: {counts['none']}, Z-score: {counts['zscore']}, "
              f"Min-Max: {counts['minmax']}, Robust: {counts['robust']}")

    def get_scaler_stats(self) -> dict:
        """Return scaler statistics for reuse in test dataset."""
        return {'center': self.center, 'scale': self.scale}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center = sample['center']
        hw = self.half_window

        # Extract symmetric window
        window = sample['feat_matrix'][center - hw:center + hw]  # (seq_len, n_features)

        # Hybrid normalize: (x - center) / scale
        # For no-scale: center=0, scale=1 (identity)
        # For minmax: center=min, scale=range
        # For robust: center=median, scale=IQR
        # For zscore: center=mean, scale=std
        window = (window - self.center) / self.scale

        # Handle any NaN in features (replace with 0 after normalization)
        window = np.nan_to_num(window, nan=0.0)

        x_tensor = torch.from_numpy(window)

        return {
            'x': x_tensor,                              # (seq_len, n_features)
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
    """
    has_target = df['radiation'].notna()
    print(f"\n[DATASET] Creating temporal CV split (val_year={val_year})...")
    print(f"  Total rows with target: {has_target.sum():,}")

    train_dataset = SolarDataset(df, feature_cols, is_train=True, scaler_stats=None)
    scaler_stats = train_dataset.get_scaler_stats()

    val_dataset = SolarDataset(df, feature_cols, is_train=False, scaler_stats=scaler_stats)

    return train_dataset, val_dataset, scaler_stats


def create_test_dataset(df: pd.DataFrame, feature_cols: list,
                        scaler_stats: dict, hparams: dict = None):
    """Create test dataset for inference."""
    return SolarDataset(df, feature_cols, is_train=False,
                        scaler_stats=scaler_stats, hparams=hparams)
