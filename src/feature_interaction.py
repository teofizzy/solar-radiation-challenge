import numpy as np
import pandas as pd
from .config import DTYPE

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes multiplicative physics interactions based on AI ensemble consensus.
    These features provide explicit inductive biases for atmospheric attenuation.
    """
    print("[INTERACTION] Computing multiplicative physics interactions...")
    
    new_cols = {}
    
    # 1. Zenith-Coupled Attenuation (Path-length scaling)
    if 'cos_zenith' in df.columns:
        # Humidity absorption scales with path length
        if 'relativehumidity' in df.columns:
            new_cols['zenith_humidity'] = (df['cos_zenith'] * df['relativehumidity']).astype(DTYPE)
        
        # Cloud attenuation scales with potential radiation
        if 'kt_landsaf' in df.columns:
            new_cols['zenith_cloud'] = (df['cos_zenith'] * df['kt_landsaf']).astype(DTYPE)
            
    # 2. Air Mass / Path-Length Scaling
    if 'air_mass' in df.columns:
        # Aerosol scattering increases with air mass
        if 'tropomi_aerosol' in df.columns:
            new_cols['airmass_aerosol'] = (df['air_mass'] * df['tropomi_aerosol']).astype(DTYPE)
            
        # Total column water vapor absorption scales with air mass
        if 'tcwv' in df.columns:
            new_cols['airmass_water'] = (df['air_mass'] * df['tcwv']).astype(DTYPE)

    # 3. Regime Detection (Differential memory)
    if 'ewma_kt_fast' in df.columns and 'ewma_kt_slow' in df.columns:
        new_cols['clearness_regime_shift'] = (df['ewma_kt_fast'] - df['ewma_kt_slow']).astype(DTYPE)

    # 4. Cloud Advection Potential (Dynamic wind-cloud interaction)
    if 'wind_speed' in df.columns and 'kt_landsaf' in df.columns:
        kt_grad = df.groupby('station')['kt_landsaf'].diff(4).fillna(0)
        new_cols['cloud_advection_potential'] = (df['wind_speed'] * kt_grad.abs()).astype(DTYPE)

    # Assign new columns
    for col, values in new_cols.items():
        df[col] = values
        
    print(f"  Added {len(new_cols)} interaction features: {list(new_cols.keys())}")
    return df
