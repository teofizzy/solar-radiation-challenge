import os
import gc
import numpy as np
import pandas as pd
from src.config import PATHS, DTYPE, DTYPE_STR
from src.utils import timer

def compute_landsaf_features(df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """
    Integrate LSA SAF satellite features (MDSSF, LST).
    MDSSF (Downward Surface Shortwave Flux) is a critical proxy for GHI.
    """
    cache_path = os.path.join(PATHS['cache_landsaf'], 'landsaf_features.parquet')

    if not force_recompute and os.path.exists(cache_path):
        print("[FEATURE_LANDSAF] Loading from cache...")
        try:
            df_saf = pd.read_parquet(cache_path)
            # Ensure index matches
            existing = [c for c in df_saf.columns if c in df.columns]
            if existing:
                df.drop(columns=existing, inplace=True)
            df = df.merge(df_saf, left_index=True, right_index=True, how='left')
            return df
        except Exception as e:
            print(f"[FEATURE_LANDSAF] Cache load failed: {e}. Recomputing...")

    with timer("FEATURE_LANDSAF"):
        print("[FEATURE_LANDSAF] Processing LSA SAF data...")

    # We need timestamp and station to match our main DataFrame
    df_saf = pd.DataFrame(index=df.index)

    # 1. MDSSF (Downward Surface Shortwave Flux)
    if os.path.exists(PATHS['mdssf_csv']):
        mdssf_raw = pd.read_csv(PATHS['mdssf_csv'])
        mdssf_raw['timestamp'] = pd.to_datetime(mdssf_raw['timestamp'])
        # Melt to long format
        mdssf_melted = mdssf_raw.melt(
            id_vars=['timestamp'], 
            var_name='station', 
            value_name='mdssf'
        )
        
        # Merge with main df (left join on timestamp and station)
        temp_df = df[['timestamp', 'station']].merge(
            mdssf_melted, on=['timestamp', 'station'], how='left'
        )
        
        # Forward fill to cover missing intervals (limit 8 steps = 2 hours)
        temp_df['mdssf'] = temp_df.groupby('station')['mdssf'].ffill(limit=8).astype(DTYPE)
        # Fill remaining with 0 (safe assumption for missing GHI mostly at night or extremely cloudy)
        df_saf['mdssf'] = temp_df['mdssf'].fillna(0.0)
    else:
        print(f"[FEATURE_LANDSAF] Warning: MDSSF file not found at {PATHS['mdssf_csv']}")
        df_saf['mdssf'] = np.float32(0.0)

    # 2. LST (Land Surface Temperature)
    if os.path.exists(PATHS['mlst_csv']):
        mlst_raw = pd.read_csv(PATHS['mlst_csv'])
        mlst_raw['timestamp'] = pd.to_datetime(mlst_raw['timestamp'])
        mlst_melted = mlst_raw.melt(
            id_vars=['timestamp'], 
            var_name='station', 
            value_name='mlst'
        )
        temp_df = df[['timestamp', 'station']].merge(
            mlst_melted, on=['timestamp', 'station'], how='left'
        )
        # Forward fill LST (changes slowly)
        temp_df['mlst'] = temp_df.groupby('station')['mlst'].ffill(limit=8).astype(DTYPE)
        
        # Impute remaining missing LST with ERA5 t2m (already in Celsius) or TAHMO temperature
        if 'temperature' in df.columns:
            fallback = df['temperature']
        elif 't2m_celsius' in df.columns:
            fallback = df['t2m_celsius']
        else:
            fallback = 25.0
            
        df_saf['mlst'] = temp_df['mlst'].fillna(fallback).astype(DTYPE)
    else:
        print(f"[FEATURE_LANDSAF] Warning: LST file not found at {PATHS['mlst_csv']}")
        df_saf['mlst'] = np.float32(0.0)

    # 3. Calculate kt_landsaf proxy
    if 'clear_sky_ghi' in df.columns:
        # Prevent division by zero
        denom = np.where(df['clear_sky_ghi'] > 1.0, df['clear_sky_ghi'], 1.0)
        df_saf['kt_landsaf'] = (df_saf['mdssf'].values / denom).astype(DTYPE)
        df_saf['kt_landsaf'] = np.clip(df_saf['kt_landsaf'], 0.0, 1.5)
    else:
        df_saf['kt_landsaf'] = np.float32(0.0)

    # Cache
    df_saf.to_parquet(cache_path)
    print(f"  Cached LandSAF features to {cache_path}")

    # Merge
    df = df.merge(df_saf, left_index=True, right_index=True, how='left')
    
    del df_saf
    if 'temp_df' in locals(): del temp_df
    if 'mdssf_raw' in locals(): del mdssf_raw
    if 'mdssf_melted' in locals(): del mdssf_melted
    if 'mlst_raw' in locals(): del mlst_raw
    if 'mlst_melted' in locals(): del mlst_melted
    gc.collect()

    return df
