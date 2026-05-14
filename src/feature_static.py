"""
Topographic geomorphometry extraction from static NetCDF priors.
Fuses DEM, Slope, Aspect (sin/cos), LandCover (OHE), and DistanceToWater.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from src.config import PATHS, DTYPE, get_station_meta, ensure_dirs
from src.utils import timer

# Static features to extract and preserve
STATIC_COLS = [
    'dem', 'slope', 'aspect_sin', 'aspect_cos', 
    'dist_water', 'tpi_2000m', 'std_2000m'
]

def compute_static_features(df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """
    Extract topographic priors for each station and merge into main DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame containing 'station' column.
    force_recompute : bool
        If True, ignores existing static columns.

    Returns
    -------
    pd.DataFrame with topographic features merged.
    """
    # Check if features already exist
    existing = [c for c in STATIC_COLS if c in df.columns]
    if len(existing) == len(STATIC_COLS) and not force_recompute:
        print("  Static features already present. Skipping.")
        return df

    static_priors_path = PATHS['static_priors']
    if not os.path.exists(static_priors_path):
        print(f"  WARNING: Static priors not found at {static_priors_path}. Skipping.")
        return df

    station_meta = get_station_meta()
    
    with timer("FEATURE_STATIC"):
        print(f"  Fusing topographic priors from {os.path.basename(static_priors_path)}...")
        
        ds = xr.open_dataset(static_priors_path)
        
        # Temporary storage for per-station features
        station_features = []
        
        if 'station' not in station_meta.columns:
            station_meta = station_meta.reset_index()
            
        for _, row in station_meta.iterrows():
            st_id = row['station']
            lat, lon = row['latitude'], row['longitude']
            
            # Spatial Extraction: Continuous gets Bilinear, Categorical gets Nearest
            point_cont = ds.interp(lat=lat, lon=lon, method='linear')
            point_near = ds.sel(lat=lat, lon=lon, method='nearest')
            
            # Core topographic variables
            dem = float(point_cont.dem.values) if 'dem' in ds else np.nan
            slope = float(point_cont.Slope.values) if 'Slope' in ds else np.nan
            aspect = float(point_cont.Aspect.values) if 'Aspect' in ds else np.nan
            
            # Robust categorical extraction
            lc_val = point_near.LandCover.values if 'LandCover' in ds else np.nan
            land_cover = int(lc_val) if not np.isnan(lc_val) else -1
            
            dist_water = float(point_cont.DistanceToWater.values) if 'DistanceToWater' in ds else np.nan
            tpi = float(point_cont.TPI_2000M.values) if 'TPI_2000M' in ds else np.nan
            std = float(point_cont.STD_2000M.values) if 'STD_2000M' in ds else np.nan
            
            # Aspect transformation (Trigonometric encoding)
            aspect_sin = np.sin(np.radians(aspect)) if not np.isnan(aspect) else 0.0
            aspect_cos = np.cos(np.radians(aspect)) if not np.isnan(aspect) else 0.0
            
            st_feat = {
                'station': st_id,
                'dem': dem,
                'slope': slope,
                'aspect_sin': aspect_sin,
                'aspect_cos': aspect_cos,
                'land_cover': land_cover,
                'dist_water': dist_water,
                'tpi_2000m': tpi,
                'std_2000m': std
            }
            station_features.append(st_feat)
        
        ds.close()
        
        # Create station-level feature dataframe
        df_static = pd.DataFrame(station_features)
        
        # Impute missing with medians
        for col in ['dem', 'slope', 'dist_water', 'tpi_2000m', 'std_2000m']:
            df_static[col] = df_static[col].fillna(df_static[col].median())
        
        # One-Hot Encode LandCover (Zindi baseline uses this)
        df_static = pd.get_dummies(df_static, columns=['land_cover'], prefix='lu')
        
        # Merge back into main dataframe
        # Ensure we don't duplicate columns if they partially exist
        cols_to_merge = [c for c in df_static.columns if c == 'station' or c not in df.columns]
        df = df.merge(df_static[cols_to_merge], on='station', how='left')
        
        # Final cleanup: ensure dtypes
        for col in df_static.columns:
            if col != 'station' and col in df.columns:
                df[col] = df[col].astype(DTYPE)

    return df
