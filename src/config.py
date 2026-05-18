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
IS_KAGGLE = os.path.exists('/kaggle')

# Base project directory
if IS_COLAB:
    PROJECT_DIR = '/content/drive/MyDrive/TAHMO_Challenge'
    LOCAL_DATA_DIR = PROJECT_DIR 
elif IS_KAGGLE:
    # On Kaggle, priority: 1. Working dir (Drive cache), 2. Input dataset
    PROJECT_DIR = '/kaggle/working/TAHMO_Challenge'
    input_dataset_path = '/kaggle/input/tahmo-solar-radiation-data'
    
    if os.path.exists(os.path.join(PROJECT_DIR, 'Train.csv')):
        LOCAL_DATA_DIR = PROJECT_DIR
    elif os.path.exists(input_dataset_path):
        LOCAL_DATA_DIR = input_dataset_path
    else:
        # Fallback to current working directory
        LOCAL_DATA_DIR = os.getcwd()
else:
    # Use SCRATCH if available, otherwise fallback to current directory
    scratch_base = os.environ.get("SCRATCH")
    if scratch_base:
        PROJECT_DIR = os.path.join(scratch_base, "challenges/zindi/solar-radiation-challenge")
    else:
        # Local development fallback: use absolute path of the directory containing this file's parent
        PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
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
    'mdssf_csv': os.path.join(LOCAL_DATA_DIR, 'mdssf.csv'),
    'mlst_csv': os.path.join(LOCAL_DATA_DIR, 'mlst.csv'),
    'tropomi_aerosol_dir': os.path.join(LOCAL_DATA_DIR, 'TROPOMI_Optimized_Aerosol'),
    'tropomi_cloud_dir': os.path.join(LOCAL_DATA_DIR, 'TROPOMI_Optimized_Cloud'),

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
    # Sequence / windowing (V1 baseline: 12h symmetric)
    'seq_len': 48,            # 12 hours @ 15-min cadence
    'half_window': 24,        # One side of the symmetric window

    # BiLSTM Architecture (direct kt prediction -- solar-sweep-1 proven)
    'hidden_dim': 160,        # BiLSTM hidden dim
    'n_layers': 2,            # 2-layer BiLSTM (locked -- 3L causes gradient explosion)
    'dropout': 0.15,          # Regularization
    'station_embed_dim': 16,  # Station embedding dimension

    # Training (FP32 -- no AMP to avoid autocast swings)
    'batch_size': 64,
    'lr': 1e-3,               # AdamW LR
    'weight_decay': 1e-4,
    'patience': 15,           # Early stopping patience
    'epochs': 80,             # Sufficient for BiLSTM convergence
    'grad_clip': 1.0,
    'use_amp': False,         # FP32 training

    # Loss: ZindiLoss (PROVEN in solar-sweep-1, Zindi=45.48)
    'lambda_smooth': 0.001,   # kt smoothness penalty in ZindiLoss (sweepable)

    # Physics & Constraints
    'kt_max': 1.05,           # Maximum clearness index (physical bound)
    'night_zenith_threshold': 90.0,
    'clearsky_min_denom': 1.0,
}

# ------------------------------------------------------------------
# 2b. STAGE 2: LightGBM Configuration (sequential residual correction)
# ------------------------------------------------------------------
STAGE2_HPARAMS = {
    'n_estimators': 420,
    'max_depth': 7,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    # Calibration bounds
    'calibration_clip': [0.8, 1.2],
}

# ------------------------------------------------------------------
# 3. W&B MLOps CONFIGURATION
# ------------------------------------------------------------------
WANDB_CONFIG = {
    'project': 'tahmo-solar-radiation',
    'entity': 'teofilo48ligawa-dsail', # Explicit team entity for CSCS
}

# ------------------------------------------------------------------
# 4. FEATURE TOGGLES & DEFINITIONS
# ------------------------------------------------------------------
FEATURES = {
    'use_era5': True,
    'use_physics': True,
    'use_temporal': True,
    'use_landsaf': True,       # Phase C
    'use_static': True,       # Phase C
    'use_tropomi': True,       # Phase D (optional)
}

# ------------------------------------------------------------------
# Pruned for memory: 1h (cloud transients), 24h (diurnal), 72h (synoptic)
MULTI_SCALE_LAGS = {
    '1h': 4,
    '24h': 96,
    '72h': 288
}

# Static Diagnostic Descriptors (from full_diagnostic.py)
DIAGNOSTIC_FEATURES = [
    'night_offset', 'drift_slope', 'avg_residual_bias', 
    'train_null_pct', 'max_rad'
]

# ------------------------------------------------------------------
# 6. ERA5 VARIABLES
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
            _STATION_META = train_df.drop_duplicates(subset='station').reset_index(drop=True)
            _STATION_META.to_csv(meta_path, index=False)
            print(f"[CONFIG] Generated and saved station_meta.csv to {meta_path}")
        else:
            _STATION_META = pd.read_csv(meta_path)
    return _STATION_META


def get_n_stations():
    """Number of unique stations."""
    return len(get_station_meta())


def ensure_dirs():
    """Create all cache and output directories."""
    for key, path in PATHS.items():
        if 'cache' in key or key in ('experiments_dir', 'submissions_dir'):
            os.makedirs(path, exist_ok=True)
