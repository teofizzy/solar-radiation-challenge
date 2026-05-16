import numpy as np
import pandas as pd
from .config import DTYPE

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes MINIMAL physics interactions based on Multi-AI consensus.
    
    Pruned features (ChatGPT, Compare AIs consensus):
      - zenith_humidity: Transformer learns cos_zenith * humidity internally
      - zenith_cloud: Same redundancy issue
      - airmass_water: Redundant with airmass_aerosol (similar Beer-Lambert physics)
      - cloud_advection_potential: Complex derived; Transformer learns wind*kt_gradient
    
    Kept features:
      - airmass_aerosol: Strong Beer-Lambert physics (scattering scales with path length)
      - clearness_regime_shift: Fast-slow EWMA differential (cloud onset/clearing detector)
    """
    print("[INTERACTION] Computing minimal physics interactions...")
    
    new_cols = {}
    
    # 1. Air Mass x Aerosol (Beer-Lambert Law: scattering scales with optical path)
    # This is the ONLY interaction that all AI sources agree should be kept.
    if 'air_mass' in df.columns and 'tropomi_aerosol' in df.columns:
        new_cols['airmass_aerosol'] = (df['air_mass'] * df['tropomi_aerosol']).astype(DTYPE)

    # 2. Regime Detection (Differential memory: fast vs slow EWMA)
    if 'ewma_kt_fast' in df.columns and 'ewma_kt_slow' in df.columns:
        new_cols['clearness_regime_shift'] = (df['ewma_kt_fast'] - df['ewma_kt_slow']).astype(DTYPE)

    # Assign new columns
    for col, values in new_cols.items():
        df[col] = values
        
    print(f"  Added {len(new_cols)} interaction features: {list(new_cols.keys())}")
    return df
