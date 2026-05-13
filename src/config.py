"""
Global configuration for the TAHMO Solar Radiation Challenge pipeline.
All paths, hyperparameters, seeds, and feature toggles are centralized here.
"""

import os
import random
import numpy as np

# ------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ------------------------------------------------------------------
SEED = 42


def seed_everything(seed: int = SEED):
    """Set all random states for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ------------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------------
# Detect environment: Colab vs local
IS_COLAB = os.path.exists('/content')

# Base project directory
if IS_COLAB:
    PROJECT_DIR = '/content/drive/MyDrive/TAHMO_Challenge'
    # Use the same directory for data as they are all in the root folder
    LOCAL_DATA_DIR = PROJECT_DIR 
else:
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data')

PATHS = {
    # Raw data
    'train': os.path.join(LOCAL_DATA_DIR, 'Train.csv'),
    'test': os.path.join(LOCAL_DATA_DIR, 'Test.csv'),
    'station_meta': os.path.join(PROJECT_DIR, 'station_meta.csv'),
    'static_priors': os.path.join(LOCAL_DATA_DIR, 'clipped_africa_static_priors.nc'),
    'era5_dir': os.path.join(LOCAL_DATA_DIR, 'era5'),
    'sample_submission': os.path.join(LOCAL_DATA_DIR, 'SampleSubmission.csv'),

    # Drive paths for satellite data (Colab only)
    'mdssf_csv': os.path.join(PROJECT_DIR, 'mdssf.csv'),
    'mlst_csv': os.path.join(PROJECT_DIR, 'mlst.csv'),
    'tropomi_aerosol_dir': os.path.join(PROJECT_DIR, 'TROPOMI_Optimized_Aerosol'),
    'tropomi_cloud_dir': os.path.join(PROJECT_DIR, 'TROPOMI_Optimized_Cloud'),

    # Cache directories (feature engineering outputs)
    'cache_dir': os.path.join(PROJECT_DIR if IS_COLAB else PROJECT_DIR, 'cache'),
    'cache_raw': os.path.join(PROJECT_DIR, 'cache', 'raw'),
    'cache_astro': os.path.join(PROJECT_DIR, 'cache', 'astro'),
    'cache_era5': os.path.join(PROJECT_DIR, 'cache', 'era5'),
    'cache_physics': os.path.join(PROJECT_DIR, 'cache', 'physics'),
    'cache_landsaf': os.path.join(PROJECT_DIR, 'cache', 'landsaf'),
    'cache_tropomi': os.path.join(PROJECT_DIR, 'cache', 'tropomi'),
    'cache_static': os.path.join(PROJECT_DIR, 'cache', 'static'),
    'cache_temporal': os.path.join(PROJECT_DIR, 'cache', 'temporal'),
    'cache_features': os.path.join(PROJECT_DIR, 'cache', 'features'),

    # Model outputs
    'experiments_dir': os.path.join(PROJECT_DIR, 'experiments'),
    'submissions_dir': os.path.join(PROJECT_DIR, 'submissions'),
}

# ------------------------------------------------------------------
# 2. MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
HPARAMS = {
    # Sequence / windowing
    'seq_len': 48,           # 24 past + 24 future = 12hr symmetric window
    'half_window': 24,       # One side of the symmetric window

    # Transformer-BiLSTM architecture
    'hidden_dim': 256,       # Scaled up for Transformer/BiLSTM capacity
    'n_layers': 2,
    'dropout': 0.15,         # Matched to deeper architecture
    'embed_dim': 16,         # Station embedding dimension

    # Attention architecture
    'use_attention': True,   # Use CenterQueryAttention vs center-only
    'attn_dim': 128,         # Attention projection dimension
    'attn_dropout': 0.10,    # Attention weight dropout (unanimous consensus)
    'attn_temperature': 2.0, # Score scaling tau (prevents collapse)

    # Training
    'lr': 3e-4,              # Reduced from 1e-3 for OneCycleLR stability
    'weight_decay': 1e-3,    # Increased from 1e-4 (Gemini recommendation)
    'batch_size': 256,       # Scaled up for T4 GPU usage
    'epochs': 80,            # Increased from 50 (longer with lower LR)
    'patience': 15,          # Increased from 8 (match longer training)
    'grad_clip': 5.0,        # Relaxed for deep transformer fusion

    # LR Scheduler (OneCycleLR)
    'scheduler': 'onecycle',       # 'onecycle' or 'cosine'
    'onecycle_pct_start': 0.12,    # Warmup fraction
    'onecycle_div_factor': 25,     # Initial LR = max_lr / div_factor
    'onecycle_final_div': 1e4,     # Final LR = max_lr / (div * final_div)

    # Loss weights
    'mbe_weight': 0.5,
    'rmse_weight': 0.5,
    'night_penalty_weight': 0.01,
    'spike_kt_threshold': 0.7,     # Upweight errors above this kt
    'spike_weight': 3.0,           # Weight multiplier for high-kt errors

    # Physics
    'kt_max': 1.5,           # Max clearness index (cloud edge enhancement)
    'night_zenith_threshold': 90.0,
    'clearsky_min_denom': 1.0,  # Prevent division by zero in kt

    # Post-processing (RTS smoother)
    'rts_q_kt': 0.0018,     # Process noise for kt state
    'rts_q_bias': 0.00008,  # Process noise for bias state
    'rts_r': 0.012,         # Observation noise
    'savgol_window': 9,     # Savitzky-Golay window (2.25 hours)
    'savgol_polyorder': 2,  # Savitzky-Golay polynomial order
}

# ------------------------------------------------------------------
# 3. W&B MLOps CONFIGURATION
# ------------------------------------------------------------------
WANDB_CONFIG = {
    'project': 'tahmo-solar-radiation',
    'entity': None,          # Use default entity
}

# ------------------------------------------------------------------
# 4. FEATURE TOGGLES & DEFINITIONS
# ------------------------------------------------------------------
FEATURES = {
    'use_era5': True,
    'use_physics': True,
    'use_temporal': True,
    'use_landsaf': True,       # Phase C
    'use_static': False,       # Phase C
    'use_tropomi': True,       # Phase D (optional)
}

# 15-min intervals: 1h=4, 3h=12, 6h=24, 12h=48
MULTI_SCALE_LAGS = {
    '1h': 4,
    '3h': 12,
    '6h': 24,
    '12h': 48
}

# ------------------------------------------------------------------
# 4. ERA5 VARIABLES
# ------------------------------------------------------------------
ERA5_VARS = ['u10', 'v10', 'd2m', 't2m', 'sp', 'tco3', 'tcwv']

# ------------------------------------------------------------------
# 5. DTYPE POLICY
# ------------------------------------------------------------------
DTYPE = np.float32
DTYPE_STR = 'float32'

# ------------------------------------------------------------------
# 6. STATION LIST (loaded lazily from station_meta.csv)
# ------------------------------------------------------------------
_STATION_META = None


def get_station_meta():
    """Lazy-load station metadata DataFrame. Generates from Train.csv if missing."""
    global _STATION_META
    if _STATION_META is None:
        import pandas as pd
        meta_path = PATHS['station_meta']
        if not os.path.exists(meta_path):
            print(f"[CONFIG] station_meta.csv not found at {meta_path}. Generating from Train.csv...")
            train_df = pd.read_csv(PATHS['train'], usecols=['station', 'latitude', 'longitude', 'elevation'])
            _STATION_META = train_df.drop_duplicates(subset='station').set_index('station')
            _STATION_META.to_csv(meta_path)
            print(f"[CONFIG] Generated and saved station_meta.csv to {meta_path}")
        else:
            _STATION_META = pd.read_csv(meta_path, index_col=0)
    return _STATION_META


def get_n_stations():
    """Number of unique stations."""
    return len(get_station_meta())


def ensure_dirs():
    """Create all cache and output directories."""
    for key, path in PATHS.items():
        if 'cache' in key or key in ('experiments_dir', 'submissions_dir'):
            os.makedirs(path, exist_ok=True)
