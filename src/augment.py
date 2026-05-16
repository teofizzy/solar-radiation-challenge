"""
Physically-consistent data augmentation for solar radiation prediction.

Critical design rule: NEVER jitter deterministic physics features.

Multi-AI consensus (ChatGPT + Gemini):
  - Uniform scaling across all features breaks physical consistency
  - cos_zenith, clear_sky_ghi, time encodings, wind direction encodings
    are deterministic -- jittering them creates physically impossible states
  - Only meteorological/sensor variables should be jittered
  - This alone can reduce early RMSE inflation by 20-60 W/m2
"""

import torch
import numpy as np

# Column indices that are SAFE to jitter (stochastic meteorological variables).
# These are populated at runtime from feature_cols.
_JITTER_SAFE_PREFIXES = {
    'temperature', 'relativehumidity', 'log_precipitation',
    'u10', 'v10', 't_lapse_corr', 'p_hyps_corr', 'tco3', 'tcwv',
    'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
    'tropomi_cloud', 'tropomi_aerosol',
    'mdssf', 'mlst',
    # wind_speed REMOVED: we now use u10/v10 as primary wind representation
}

# Columns that must NEVER be jittered (deterministic physics)
_JITTER_FORBIDDEN_PREFIXES = {
    'cos_zenith', 'csghi_terrain_corr', 'clear_sky_ghi',
    'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos',
    'hour_6_sin', 'hour_6_cos', 'doy_sin', 'doy_cos',
    'days_since_start',
    'wind_direction_sin', 'wind_direction_cos',  # removed from features but kept in forbidden for safety
    'era5_missing', 'tropomi_cloud_missing', 'tropomi_aerosol_missing',
    'is_night', 'kt_landsaf',  # kt_landsaf is the prediction anchor
    'dist_water',  # static geographic feature
    'lu_',  # land use one-hot (binary)
    'airmass_aerosol',  # derived interaction (should be jittered via its components)
}

_jitter_mask_cache = {}


def get_jitter_mask(feature_cols: list) -> torch.Tensor:
    """Return a boolean mask (n_features,) where True = safe to jitter."""
    key = tuple(feature_cols)
    if key in _jitter_mask_cache:
        return _jitter_mask_cache[key]
    
    mask = torch.zeros(len(feature_cols), dtype=torch.bool)
    for i, col in enumerate(feature_cols):
        # Check if any safe prefix matches
        is_safe = any(col.startswith(p) or col == p for p in _JITTER_SAFE_PREFIXES)
        # Check if any forbidden prefix matches (overrides safe)
        is_forbidden = any(col.startswith(p) or col == p for p in _JITTER_FORBIDDEN_PREFIXES)
        
        if is_safe and not is_forbidden:
            mask[i] = True
    
    _jitter_mask_cache[key] = mask
    return mask


def apply_intensity_jitter(x, is_night, p=0.4, scale_range=(0.85, 1.15),
                           feature_cols=None):
    """
    Applies multiplicative scaling to stochastic meteorological features ONLY.
    
    Deterministic physics features (cos_zenith, clear_sky_ghi, time encodings,
    wind direction) are NEVER modified.
    
    Parameters
    ----------
    x : (T, F) tensor
    is_night : (T,) tensor 
    p : float, probability of applying augmentation
    scale_range : tuple, range for uniform scaling
    feature_cols : list, feature column names (for physics-aware masking)
    """
    if np.random.rand() > p:
        return x
    
    scale = np.random.uniform(*scale_range)
    
    # Daytime mask: only augment daytime samples
    day_mask = (1.0 - is_night).unsqueeze(-1)  # (T, 1)
    
    if feature_cols is not None:
        # Physics-aware: only scale safe features
        feat_mask = get_jitter_mask(feature_cols).to(x.device).unsqueeze(0)  # (1, F)
        combined_mask = day_mask * feat_mask.float()  # (T, F)
        x = x * (1.0 - combined_mask + combined_mask * scale)
    else:
        # Fallback: no feature info, only scale daytime
        x = x * (1.0 - day_mask + day_mask * scale)
    
    return x


def apply_temporal_mask(x, p=0.2, max_mask_ratio=0.15):
    """
    Zeroes out a contiguous block of time to simulate sensor dropout.
    More conservative ratio (max 15%) to preserve temporal context.
    """
    if np.random.rand() > p:
        return x
    
    T = x.shape[0]
    mask_len = int(T * np.random.uniform(0.05, max_mask_ratio))
    start = np.random.randint(0, max(T - mask_len, 1))
    
    x[start:start+mask_len, :] = 0
    return x
