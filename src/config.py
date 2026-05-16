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
    # Sequence / windowing
    'seq_len': 192,           # 48 hours @ 15-min
    'half_window': 96,        # One side of the symmetric window
    
    # Patch-Transformer Architecture
    'hidden_dim': 128,        # d_model (must be divisible by transformer_heads)
    'n_layers': 4,            # Number of Transformer layers
    'transformer_heads': 8,   # Must divide hidden_dim evenly
    'dropout': 0.15,          # Slightly higher for regularization
    'patch_len': 16,          # 4-hour patches
    'stride': 8,              # 2-hour sliding stride
    'station_embed_dim': 32,  # Latent projection of diagnostic features
    
    # Training
    'batch_size': 32,
    'lr': 3e-4,               # Slightly higher LR for single-objective loss
    'weight_decay': 1e-4,
    'patience': 15,           # Increased patience for slower convergence
    'epochs': 100,
    'grad_clip': 1.0,

    # Physics & Constraints
    'kt_max': 1.05,           # Physical kt upper bound (tightened from 1.5)
    'night_zenith_threshold': 90.0,
    'clearsky_min_denom': 1.0,
    
    # Phase 2: Curriculum Learning (clear-sky -> clouds -> all)
    # Reference: Perplexity + Deep Search consensus: 3-8 W/m2 RMSE reduction
    'use_curriculum': True,
    'curriculum_warmup_frac': 0.30,   # Epochs 0-30%: clear-sky emphasis
    'curriculum_medium_frac': 0.60,   # Epochs 30-60%: broken clouds
    
    # Phase 3: Loss Annealing (MSE -> Huber at switch_frac) + MBE anchor
    # Reference: ChatGPT: "Apply MBE in GHI space. lambda=0.005-0.02, start at 0.01"
    'huber_delta_kt': 0.03,           # delta in kt-space (~20 W/m2 at noon)
    'huber_switch_frac': 0.60,        # Switch from MSE to Huber at 60% epochs
    'lambda_mbe': 0.01,               # GHI-space MBE anchor weight (prevents bias drift)
    
    # Phase 4: Stochastic Weight Averaging (SWA)
    # Reference: ChatGPT consensus: 2-5 W/m2 RMSE reduction
    'use_swa': True,
    'swa_lr_ratio': 0.1,              # SWA LR = lr * 0.1
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
