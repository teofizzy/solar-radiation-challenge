import os
import gc
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from src.config import PATHS, DTYPE, get_station_meta
from src.utils import timer

def extract_daily_tropomi(tif_dir: str, prefix: str, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts daily TROPOMI values using a 3x3 spatial window mean.
    Returns a DataFrame indexed by (date, station) with the feature value.
    """
    files = glob.glob(os.path.join(tif_dir, f"*{prefix}*.tif"))
    print(f"    Found {len(files)} {prefix} files.")
    
    # Dictionary to hold {date: {station: [values]}}
    daily_data = {}
    
    date_pattern = re.compile(r'_(\d{8})\.tif')
    
    for f in files:
        match = date_pattern.search(f)
        if not match:
            continue
        date_str = match.group(1)
        date = pd.to_datetime(date_str, format='%Y%m%d')
        
        if date not in daily_data:
            daily_data[date] = {station: [] for station in meta.index}
            
        with rasterio.open(f) as src:
            for station, row in meta.iterrows():
                try:
                    # Get pixel row/col
                    py, px = src.index(row.longitude, row.latitude)
                    
                    # Read 3x3 window around the pixel
                    window = Window(px - 1, py - 1, 3, 3)
                    data = src.read(1, window=window)
                    
                    if data.size > 0:
                        # Mask out nodata
                        nodata = src.nodata
                        if nodata is not None:
                            valid_data = data[data != nodata]
                        else:
                            # TROPOMI often uses negative values for missing or np.nan
                            valid_data = data[~np.isnan(data)]
                            valid_data = valid_data[valid_data > -9999]
                            
                        if len(valid_data) > 0:
                            mean_val = np.mean(valid_data)
                            daily_data[date][station].append(mean_val)
                except Exception:
                    # Station outside raster bounds
                    pass
                    
    # Aggregate (mean across clusters if multiple overlap)
    records = []
    for date, st_dict in daily_data.items():
        for station, vals in st_dict.items():
            if vals:
                records.append({
                    'date': date,
                    'station': station,
                    f'tropomi_{prefix.lower()}': np.mean(vals)
                })
                
    df_extracted = pd.DataFrame(records)
    if len(df_extracted) > 0:
        df_extracted.set_index(['date', 'station'], inplace=True)
    return df_extracted

def compute_tropomi_features(df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """
    Integrate TROPOMI satellite features (Cloud, Aerosol).
    Uses scientifically correct stepwise persistence instead of interpolation.
    """
    cache_path = os.path.join(PATHS['cache_tropomi'], 'tropomi_features.parquet')

    if not force_recompute and os.path.exists(cache_path):
        print("[FEATURE_TROPOMI] Loading from cache...")
        try:
            df_tropo = pd.read_parquet(cache_path)
            existing = [c for c in df_tropo.columns if c in df.columns]
            if existing:
                df.drop(columns=existing, inplace=True)
            df = df.merge(df_tropo, left_index=True, right_index=True, how='left')
            return df
        except Exception as e:
            print(f"[FEATURE_TROPOMI] Cache load failed: {e}. Recomputing...")

    with timer("FEATURE_TROPOMI"):
        print("[FEATURE_TROPOMI] Processing TROPOMI geospatial data...")
        meta = get_station_meta()

        # 1. Extract daily data
        if os.path.exists(PATHS['tropomi_cloud_dir']):
            df_cloud = extract_daily_tropomi(PATHS['tropomi_cloud_dir'], 'CLOUD', meta)
        else:
            df_cloud = pd.DataFrame()
            
        if os.path.exists(PATHS['tropomi_aerosol_dir']):
            df_aero = extract_daily_tropomi(PATHS['tropomi_aerosol_dir'], 'AEROSOL', meta)
        else:
            df_aero = pd.DataFrame()

        # 2. Map to 15-min meteorological DataFrame
        df_tropo = pd.DataFrame(index=df.index)
        df_tropo['date'] = df['timestamp'].dt.normalize()
        df_tropo['station'] = df['station']

        # 3. Merge daily values (will be NaN for days without overpass)
        #    Guard: only merge if extracted DataFrame is non-empty AND has the
        #    expected MultiIndex (date, station). Otherwise skip gracefully.
        temp_df = df_tropo.reset_index()

        if len(df_cloud) > 0 and isinstance(df_cloud.index, pd.MultiIndex):
            temp_df = temp_df.merge(
                df_cloud, left_on=['date', 'station'], right_index=True, how='left'
            )
        else:
            # No cloud data: add placeholder column
            for col in [c for c in df_cloud.columns] if len(df_cloud) > 0 else []:
                temp_df[col] = np.nan

        if len(df_aero) > 0 and isinstance(df_aero.index, pd.MultiIndex):
            temp_df = temp_df.merge(
                df_aero, left_on=['date', 'station'], right_index=True, how='left'
            )

        temp_df.set_index('index', inplace=True)
        
        # 4. AI STRATEGY: Stepwise Persistence + Missing Masks + Age
        # Forward fill limit = 96 (24 hours at 15-min intervals)
        for var in ['cloud', 'aerosol']:
            col_name = f'tropomi_{var}'
            if col_name in temp_df.columns:
                # Create missing mask FIRST (1 if missing, 0 if present)
                is_observed = temp_df[col_name].notna()
                
                # Forward fill
                filled = temp_df.groupby('station')[col_name].ffill(limit=96).astype(DTYPE)
                
                # Fill remaining with 0
                df_tropo[col_name] = filled.fillna(0.0)
                
                # Missing mask
                df_tropo[f'{col_name}_missing'] = filled.isna().astype(DTYPE)
                
                # Age hours feature
                obs_times = df['timestamp'].copy()
                obs_times[~is_observed] = pd.NaT
                last_obs_time = obs_times.groupby(df['station']).ffill(limit=96)
                
                # Compute age in hours
                age_hours = (df['timestamp'] - last_obs_time).dt.total_seconds() / 3600.0
                df_tropo[f'{col_name}_age_hours'] = age_hours.fillna(24.0).astype(DTYPE)
                
            else:
                df_tropo[col_name] = np.float32(0.0)
                df_tropo[f'{col_name}_missing'] = np.float32(1.0)
                df_tropo[f'{col_name}_age_hours'] = np.float32(24.0)

        # Drop intermediate columns
        df_tropo.drop(columns=['date', 'station'], inplace=True)

        # Cache
        os.makedirs(PATHS['cache_tropomi'], exist_ok=True)
        df_tropo.to_parquet(cache_path)
        print(f"  Cached TROPOMI features to {cache_path}")

    # Merge into main df
    df = df.merge(df_tropo, left_index=True, right_index=True, how='left')
    
    del df_tropo
    if 'temp_df' in locals(): del temp_df
    if 'df_cloud' in locals(): del df_cloud
    if 'df_aero' in locals(): del df_aero
    gc.collect()

    return df
