
import torch
import numpy as np

def apply_intensity_jitter(x, is_night, p=0.5, scale_range=(0.7, 1.1)):
    """
    Applies multiplicative scaling to daytime GHI/kt values.
    x: (T, F) - assuming F=0 is some form of GHI or kt
    """
    if np.random.rand() > p:
        return x
    
    scale = np.random.uniform(*scale_range)
    # We only scale features that are radiation-related (usually first few indices)
    # For now, let's assume we scale the entire feature vector if it's relative
    # but strictly only for daytime points to avoid scaling nighttime noise.
    mask = (1.0 - is_night).unsqueeze(-1) # (T, 1)
    
    # Scale radiation features (e.g., first index if it's GHI/kt)
    # This is a simplified version; in reality, we'd target specific columns.
    x = x * (1.0 - mask + mask * scale)
    return x

def apply_station_dropout(station_idx, memory_size=40, p=0.2):
    """
    Returns a mask for spatial attention.
    """
    # This is actually better handled in the model's forward pass 
    # or by passing a mask to the spatial attention.
    pass

def apply_temporal_mask(x, p=0.3, max_mask_ratio=0.2):
    """
    Zeroes out a contiguous block of time to simulate sensor failure.
    """
    if np.random.rand() > p:
        return x
    
    T = x.shape[0]
    mask_len = int(T * np.random.uniform(0.05, max_mask_ratio))
    start = np.random.randint(0, T - mask_len)
    
    x[start:start+mask_len, :] = 0
    return x
