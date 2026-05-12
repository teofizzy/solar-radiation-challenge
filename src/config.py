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
    LOCAL_DATA_DIR = '/content/data'
else:
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data')

PATHS = {
    # Raw data
    'train': os.path.join(LOCAL_DATA_DIR, 'Train.csv'),
    'test': os.path.join(LOCAL_DATA_DIR, 'Test.csv'),
    'station_meta': os.path.join(
        PROJECT_DIR if not IS_COLAB else '/content', 'station_meta.csv'
    ),
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

    # BiLSTM architecture
    'hidden_dim': 128,
    'n_layers': 2,
    'dropout': 0.2,
    'embed_dim': 16,         # Station embedding dimension

    # Training
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'epochs': 50,
    'patience': 8,           # Early stopping patience
    'grad_clip': 1.0,

    # Loss weights
    'mbe_weight': 0.5,
    'rmse_weight': 0.5,
    'smoothness_weight': 0.02,
    'night_penalty_weight': 0.01,

    # Physics
    'kt_max': 1.5,           # Max clearness index (cloud edge enhancement)
    'night_zenith_threshold': 90.0,
    'clearsky_min_denom': 1.0,  # Prevent division by zero in kt
}

# ------------------------------------------------------------------
# 3. FEATURE TOGGLES
# ------------------------------------------------------------------
FEATURES = {
    'use_era5': True,
    'use_physics': True,
    'use_temporal': True,
    'use_landsaf': False,      # Phase C
    'use_static': False,       # Phase C
    'use_tropomi': False,      # Phase D (optional)
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
    """Lazy-load station metadata DataFrame."""
    global _STATION_META
    if _STATION_META is None:
        import pandas as pd
        meta_path = PATHS['station_meta']
        if not os.path.exists(meta_path):
            # Fallback for Colab where station_meta might be in project root
            alt_path = os.path.join(PROJECT_DIR, 'station_meta.csv')
            if os.path.exists(alt_path):
                meta_path = alt_path
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
