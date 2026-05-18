"""
PyTorch Dataset for the Hybrid BiLSTM + LightGBM Pipeline.
Symmetric sliding windows for retrospective GHI reconstruction.

Architecture (Hybrid V2 -- multi-AI consensus):
- SYMMETRIC windows: 24 past + 24 future = 48 steps (12hr at 15-min cadence)
- Target: raw residual (GHI_true - MDSSF) clipped to [-200, 200] W/m2
- BiLSTM gets 34 lean features (no rolling/lag/EWMA)
- LightGBM gets ALL features including rolling/lag/EWMA/RTS
- Hybrid normalization: No-scale / Min-Max / Robust / Z-score by family

References:
  - Multi-AI consensus: "BiLSTM learns temporal patterns from sequence --
    rolling stats are redundant but useful for LightGBM"
  - PISSM paper: Z-score for meteorological, no-scale for SZA
  - Perplexity: "Residual learning from satellite is unequivocally superior"
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


# ======================================================================
# BILSTM FEATURES: 34 lean core features (no rolling/lag/EWMA)
# ======================================================================
# These are the features the BiLSTM sees. Rolling/lag/EWMA are redundant
# for a sequence model but will be passed to LightGBM in Stage 2.

ASTRO_FEATURES = ['cos_zenith', 'csghi_terrain_corr']

ERA5_FEATURES = ['u10', 'v10', 't_lapse_corr', 'p_hyps_corr', 'tco3', 'tcwv', 'era5_missing']

PHYSICS_FEATURES = [
    'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
    'hour_sin', 'hour_cos',
    'days_since_start'
]

LOCAL_FEATURES = ['temperature', 'relativehumidity', 'log_precipitation']

SATELLITE_FEATURES = ['mdssf', 'kt_landsaf', 'tropomi_cloud', 'tropomi_aerosol']

STATIC_FEATURES = ['dist_water']

DRIFT_FEATURES = ['drift_proxy', 'hours_since_wash']

INTERACTION_FEATURES = ['airmass_aerosol']

# Residual clipping bounds (multi-AI consensus: [-200, 200])
RESIDUAL_CLIP_MIN = -200.0
RESIDUAL_CLIP_MAX = 200.0


# Beyond-window features: encode temporal context the 12h BiLSTM cannot compute.
# These are pre-computed rolling/lag statistics that span beyond seq_len=48.
# Evidence: ChatGPT + NotebookLM agree these provide complementary context;
# Gemini agrees only for features truly beyond the receptive field.
BEYOND_WINDOW_FEATURES = [
    'rolling_24h_cloud_std',    # Cloud variability over 24h (beyond 12h window)
    'rolling_72h_kt_anomaly',   # Clearness index anomaly vs 3-day mean
    'ewma_kt_24h',              # Exponential drift tracking over 24h
    'mdssf_lag_4',              # Satellite lag at 1h (recent cloud movement)
    'kt_landsaf_lag_4',         # Clearness index lag at 1h
    'volatility_index',         # Pre-computed variability metric
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return BiLSTM feature set, controlled by HPARAMS['use_lean_features'].

    FULL mode (default, Ablation 1+2):
    - ~109 features (proven solar-sweep-1 config)
    - Includes lu_* OHE, rolling/lag, EWMA drift features
    - Matches the config that produced Zindi=43.17

    LEAN mode (Ablation 4, use_lean_features=True):
    - ~32 features (4-source evidence-backed)
    - Excludes lu_* OHE (redundant with station embedding)
    - Excludes rolling/lag/EWMA (redundant for sequence models)
    - Includes 6 selective beyond-window features (>12h context)
    - Maintains ~8x hidden:feature ratio at hidden_dim=256
    """
    from src.config import HPARAMS
    use_lean = HPARAMS.get('use_lean_features', False)

    if use_lean:
        # Lean set: no lu_*, no rolling stats, only selective beyond-window
        candidates = (ASTRO_FEATURES + ERA5_FEATURES + PHYSICS_FEATURES +
                      LOCAL_FEATURES + SATELLITE_FEATURES + STATIC_FEATURES +
                      DRIFT_FEATURES + INTERACTION_FEATURES +
                      BEYOND_WINDOW_FEATURES)
        # lu_* OHE deliberately excluded (see comments in BEYOND_WINDOW_FEATURES)
    else:
        # Full set: include everything (matches solar-sweep-1)
        candidates = (ASTRO_FEATURES + ERA5_FEATURES + PHYSICS_FEATURES +
                      LOCAL_FEATURES + SATELLITE_FEATURES + STATIC_FEATURES +
                      DRIFT_FEATURES + INTERACTION_FEATURES)

        # Land Use OHE (static per station)
        lu_cols = sorted([c for c in df.columns if c.startswith('lu_')])
        candidates = list(candidates) + lu_cols

        # Rolling/lag/diff features
        rolling_cols = sorted([c for c in df.columns if '_roll_' in c or '_diff_' in c or
                               c in ('volatility_index', 'sticky_dust_index', 'advection_kt')])
        candidates = candidates + rolling_cols

        # Satellite lag stacks
        lag_cols = sorted([c for c in df.columns if '_lag_' in c])
        candidates = candidates + lag_cols

        # EWMA drift features
        ewma_cols = sorted([c for c in df.columns
                            if c.startswith('ewma_kt_') or
                            c in ('log_cum_exposure', 'clearness_regime_shift', 'ewma_residual_kt')])
        candidates = candidates + ewma_cols

    # Filter to available and deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c in df.columns and c not in seen:
            seen.add(c)
            unique.append(c)

    mode_tag = "LEAN" if use_lean else "FULL"
    print(f"[DATASET] Feature set: {mode_tag} ({len(unique)} features)")
    return unique


def get_lgbm_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the FULL feature set for LightGBM Stage 2 (109+ features).
    Includes rolling/lag/EWMA/RTS that are useful for tree models.
    """
    # Start with BiLSTM core features
    candidates = list(get_feature_columns(df))

    # Extra satellite: mlst (redundant for BiLSTM but useful for trees)
    extra_satellite = ['mlst', 'tropomi_cloud_missing', 'tropomi_cloud_age_hours',
                       'tropomi_aerosol_missing', 'tropomi_aerosol_age_hours']
    candidates += [c for c in extra_satellite if c in df.columns]

    # Rolling/lag/diff/std features (valuable for trees)
    rolling_cols = sorted([c for c in df.columns if '_roll_' in c or '_diff_' in c or
                           c in ('volatility_index', 'sticky_dust_index', 'advection_kt')])
    candidates += rolling_cols

    # Satellite lag stacks (e.g., mdssf_lag_1, kt_landsaf_lag_4)
    lag_cols = sorted([c for c in df.columns if '_lag_' in c])
    candidates += lag_cols

    # EWMA drift features
    ewma_cols = sorted([c for c in df.columns
                        if c.startswith('ewma_kt_') or
                        c in ('log_cum_exposure', 'clearness_regime_shift')])
    candidates += ewma_cols

    # RTS smoother features (if available -- added by postprocess)
    rts_cols = ['rts_estimate', 'rts_residual']
    candidates += [c for c in rts_cols if c in df.columns]

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
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
        mdssf_ghi: float -- MDSSF satellite baseline at center timestep
        clear_sky_ghi: float -- clear-sky GHI at center timestep (for clamping)
        is_night: float -- nighttime flag at center timestep
        target_residual: float -- clipped (radiation - mdssf) at center (NaN for test)
        target_ghi: float -- radiation at center (NaN for test)
    """

    # ---- Feature categories for hybrid normalization ----
    _NO_SCALE_PREFIXES = {
        'cos_zenith', 'kt_landsaf',
        'hour_sin', 'hour_cos',
        'era5_missing', 'tropomi_cloud_missing', 'tropomi_aerosol_missing',
        'is_night',
        'lu_',
    }
    _MINMAX_FEATURES = {
        'csghi_terrain_corr', 'days_since_start',
        'mdssf', 'mlst',
        'dist_water',
    }
    _ROBUST_FEATURES = {
        'log_precipitation', 'tropomi_aerosol',
    }
    # Everything else: Z-score

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 is_train: bool = True, scaler_stats: dict = None, hparams: dict = None):

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

            # MDSSF satellite baseline (needed for residual target + model forward)
            if 'mdssf' in st_df.columns:
                mdssf = st_df['mdssf'].values.astype(np.float32)
            else:
                mdssf = np.zeros(n_rows, dtype=np.float32)

            # Raw radiation target
            if 'radiation' in st_df.columns:
                target_ghi = st_df['radiation'].values.astype(np.float32)
            else:
                target_ghi = np.full(n_rows, np.nan, dtype=np.float32)

            # Residual target: GHI_true - MDSSF, clipped to [-200, 200]
            target_residual = target_ghi - mdssf
            target_residual = np.clip(target_residual, RESIDUAL_CLIP_MIN, RESIDUAL_CLIP_MAX)

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
                    'mdssf_ghi': mdssf[center],
                    'clear_sky_ghi': clear_sky[center],
                    'is_night': is_night[center],
                    'target_residual': target_residual[center],
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
        """Classify a feature column into its normalization category."""
        for prefix in cls._NO_SCALE_PREFIXES:
            if col == prefix or col.startswith(prefix):
                return 'none'
        if col in cls._MINMAX_FEATURES or any(col.startswith(p) for p in cls._MINMAX_FEATURES):
            return 'minmax'
        if col in cls._ROBUST_FEATURES:
            return 'robust'
        return 'zscore'

    def _compute_scaler(self, df: pd.DataFrame, feature_cols: list):
        """
        Physics-aware hybrid normalization.

        All statistics computed on TRAINING months only (odd months).
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
                self.center[i] = 0.0
                self.scale[i] = 1.0

            elif cat == 'minmax':
                c_min = col_data.min()
                c_max = col_data.max()
                self.center[i] = np.float32(c_min)
                self.scale[i] = np.float32(max(c_max - c_min, 1e-6))

            elif cat == 'robust':
                self.center[i] = np.float32(col_data.median())
                q25 = np.float32(col_data.quantile(0.25))
                q75 = np.float32(col_data.quantile(0.75))
                iqr = q75 - q25
                self.scale[i] = np.float32(max(iqr, 1e-6))

            else:  # zscore
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
        window = (window - self.center) / self.scale

        # Handle any NaN in features (replace with 0 after normalization)
        window = np.nan_to_num(window, nan=0.0)

        x_tensor = torch.from_numpy(window)

        return {
            'x': x_tensor,                              # (seq_len, n_features)
            'station_idx': torch.tensor(sample['station_idx'], dtype=torch.long),
            'mdssf_ghi': torch.tensor(sample['mdssf_ghi'], dtype=torch.float32),
            'clear_sky_ghi': torch.tensor(sample['clear_sky_ghi'], dtype=torch.float32),
            'is_night': torch.tensor(sample['is_night'], dtype=torch.float32),
            'target_residual': torch.tensor(sample['target_residual'], dtype=torch.float32),
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
