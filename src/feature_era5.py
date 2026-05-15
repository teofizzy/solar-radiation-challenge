"""
ERA5 reanalysis feature extraction with PCHIP interpolation.
Processes month-by-month to avoid RAM blowups on Colab.
Interpolates hourly ERA5 data to 15-minute TAHMO cadence.
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import PchipInterpolator

from src.config import PATHS, ERA5_VARS, DTYPE, ensure_dirs
from src.utils import timer, reduce_mem_usage, validate_no_nan, clean_memory


def normalize_era5_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize coordinate names for spatial (lat/lon) and temporal (time) axes.
    Ensures 'latitude', 'longitude', and 'time' are present if available.
    """
    rename_map = {}
    
    # Spatial coordinates
    if 'lat' in ds.coords and 'latitude' not in ds.coords:
        rename_map['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords:
        rename_map['lon'] = 'longitude'
        
    # Temporal coordinates
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        rename_map['valid_time'] = 'time'
    elif 'forecast_time' in ds.coords and 'time' not in ds.coords:
        rename_map['forecast_time'] = 'time'
    
    if rename_map:
        print(f"  [FEATURE_ERA5] Normalizing coordinates: {rename_map}")
        ds = ds.rename(rename_map)
        
    # Ensure time is 1D if it was expanded
    if 'time' in ds.coords and ds['time'].ndim > 1:
        ds = ds.isel(step=0, drop=True) if 'step' in ds.dims else ds
        
    return ds


def extract_era5_for_station(ds: xr.Dataset, station_id: str, lat: float, lon: float,
                             target_timestamps: pd.Series) -> pd.DataFrame:
    """
    Extract ERA5 variables for a single station and interpolate to 15-min cadence.
    """
    # 1. Robust Longitude Normalization
    ds_lon_min = float(ds.longitude.min())
    ds_lon_max = float(ds.longitude.max())
    
    if ds_lon_min >= 0 and lon < 0:
        lon = lon % 360
    elif ds_lon_max <= 180 and lon > 180:
        lon = (lon + 180) % 360 - 180

    # 2. Standardize naming and FORCE SORTING
    ds = normalize_era5_coords(ds)
    ds = ds.sortby(['latitude', 'longitude'])
    
    # 3. Robust Temporal Alignment
    # Ensure target_timestamps are naive UTC
    target_timestamps = pd.to_datetime(target_timestamps)
    if hasattr(target_timestamps, 'dt'):
        target_timestamps = target_timestamps.dt.tz_localize(None)
    else:
        target_timestamps = target_timestamps.tz_localize(None)
    
    try:
        # A. Spatial selection FIRST (nearest)
        ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
        
        # B. Handle variables and expver
        available_vars = [v for v in ERA5_VARS if v in ds_point.data_vars]
        if not available_vars:
            return pd.DataFrame(index=target_timestamps)
        
        ds_point = ds_point[available_vars]
        if 'expver' in ds_point.dims:
            ds_point = ds_point.sel(expver=1).combine_first(ds_point.sel(expver=5))
            
        # C. Compute and convert to DataFrame
        df_era = ds_point.compute().to_dataframe()
        
        # D. Ensure DatetimeIndex
        if isinstance(df_era.index, pd.MultiIndex):
            df_era = df_era.reset_index()
            time_col = [c for c in df_era.columns if 'time' in c.lower() or c == 'valid_time'][0]
            df_era = df_era.set_index(time_col)
        
        df_era.index = pd.to_datetime(df_era.index).tz_localize(None)
        # Remove duplicates and sort
        df_era = df_era[~df_era.index.duplicated(keep='first')].sort_index()
        
        # E. Reindex and Interpolate
        # Unionize original ERA5 timestamps with target timestamps to ensure PCHIP has anchor points
        all_timestamps = df_era.index.union(target_timestamps).sort_values()
        df_full = df_era[available_vars].reindex(all_timestamps)
        
        # Interpolate across the full unionized index
        df_full = df_full.interpolate(method='pchip', limit_direction='both')
        
        # Select ONLY the target timestamps
        df_final = df_full.loc[target_timestamps]
        
        # Final safety: fill any remaining NaNs at edges
        df_final = df_final.ffill().bfill()
        
        # F. Terrain Corrections (Lapse Rate & Hypsometry)
        # Note: These are now correctly applied in feature_physics.py to avoid duplication
        # and to ensure they run after all raw features are extracted.

        # Indicator for successfully extracted data
        df_final['era5_missing'] = np.float32(0.0)
        
        # Ensure 'timestamp' is a column for merging
        df_final = df_final.reset_index(names='timestamp')
        
        return df_final.astype({c: DTYPE for c in df_final.columns if c != 'timestamp'})

    except Exception as e:
        print(f"  [FEATURE_ERA5] Error extracting {station_id}: {e}")
        # Structured fallback: return NaNs but preserve timestamp column and add missing flag
        fallback = pd.DataFrame({
            'timestamp': target_timestamps,
            'era5_missing': np.float32(1.0)
        })
        for var in ERA5_VARS:
            fallback[var] = np.float32(np.nan)
        return fallback.astype({c: DTYPE for c in fallback.columns if c != 'timestamp'})


def compute_era5_features(df: pd.DataFrame,
                          force_recompute: bool = False) -> pd.DataFrame:
    """
    Extract and interpolate ERA5 features for all stations, year-by-year.

    Processes one year at a time to stay within Colab RAM limits.
    Caches per station-year parquet files.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with timestamp, station, latitude, longitude.
    force_recompute : bool
        If True, recompute even if cache exists.

    Returns
    -------
    pd.DataFrame with ERA5 features merged.
    """
    ensure_dirs()
    era5_dir = PATHS['era5_dir']

    # Check if ERA5 data exists
    if not os.path.exists(era5_dir):
        print("[FEATURE_ERA5] WARNING: ERA5 directory not found. Skipping.")
        for var in ERA5_VARS:
            df[var] = np.float32(np.nan)
        return df

    with timer("FEATURE_ERA5"):
        print("[FEATURE_ERA5] Processing ERA5 reanalysis data...")

        stations = sorted(df['station'].unique())
        years = sorted(df['year'].unique())

        all_era5_dfs = []

        for year in years:
            year_dir = os.path.join(era5_dir, str(year))
            if not os.path.exists(year_dir):
                print(f"  WARNING: No ERA5 data for {year}")
                continue

            # Check if all station-year caches exist
            all_cached = True
            for station_id in stations:
                cache_path = os.path.join(
                    PATHS['cache_era5'], f'{station_id}_{year}.parquet')
                if not os.path.exists(cache_path) or force_recompute:
                    all_cached = False
                    break

            if all_cached and not force_recompute:
                print(f"  {year}: all stations cached, loading...")
                year_valid = True
                from src.utils import enforce_schema
                for station_id in stations:
                    cache_path = os.path.join(PATHS['cache_era5'], f'{station_id}_{year}.parquet')
                    cached_df = pd.read_parquet(cache_path)
                    try:
                        cached_df = enforce_schema(cached_df, source_name=f"ERA5_FAST_{station_id}_{year}")
                        all_era5_dfs.append(cached_df)
                    except Exception as e:
                        print(f"  [FEATURE_ERA5] Cache invalid for {station_id}_{year}: {e}. Forcing recompute.")
                        year_valid = False
                        break
                if year_valid:
                    continue
                # If invalid, fall through to full recompute for this year
                # Filter out the invalid ones that might have just been appended
                all_era5_dfs = [d for d in all_era5_dfs if d['year'].iloc[0] != year]

            # Load ERA5 for this year
            print(f"  {year}: loading ERA5 NetCDF files...")
            nc_files = sorted(glob.glob(os.path.join(year_dir, '*.nc')))
            if not nc_files:
                print(f"  WARNING: No .nc files in {year_dir}")
                continue

            # Open without specific chunks first to avoid error if 'time' is missing
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            
            # Normalize coordinates globally
            ds = normalize_era5_coords(ds)
            
            # Now safe to chunk
            if 'time' in ds.dims:
                ds = ds.chunk({'time': 100})
            
            # Diagnostic for coordinate and time ranges
            if 'latitude' in ds.coords and 'time' in ds.coords:
                lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
                lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
                t_min, t_max = pd.to_datetime(ds.time.min().values), pd.to_datetime(ds.time.max().values)
                print(f"  [FEATURE_ERA5] Bounds: Lat=[{lat_min:.2f}, {lat_max:.2f}], "
                      f"Lon=[{lon_min:.2f}, {lon_max:.2f}]")
                print(f"  [FEATURE_ERA5] Time: {t_min} to {t_max}")

            # Handle expver dimension
            if 'expver' in ds.dims:
                ds = ds.mean(dim='expver', skipna=True)

            # Process each station
            for station_id in stations:
                cache_path = os.path.join(
                    PATHS['cache_era5'], f'{station_id}_{year}.parquet')

                if os.path.exists(cache_path) and not force_recompute:
                    cached_df = pd.read_parquet(cache_path)
                    # Recovery: Parquet sometimes loses the column name or saves as index
                    from src.utils import enforce_schema
                    try:
                        cached_df = enforce_schema(cached_df, source_name=f"ERA5_CACHE_{station_id}_{year}")
                        all_era5_dfs.append(cached_df)
                        continue
                    except Exception as e:
                        print(f"  [FEATURE_ERA5] Cache corrupted for {station_id}_{year}: {e}. Recomputing...")
                        # Fall through to recompute

                # Get station coordinates
                st_mask = (df['station'] == station_id) & (df['year'] == year)
                st_data = df.loc[st_mask]
                if len(st_data) == 0:
                    continue

                lat = st_data['latitude'].iloc[0]
                lon = st_data['longitude'].iloc[0]
                target_ts = st_data['timestamp']

                # Extract and interpolate
                result = extract_era5_for_station(
                    ds, station_id, lat, lon, target_ts)

                if len(result) > 0:
                    result['station'] = station_id
                    result = reduce_mem_usage(result, verbose=False)

                    # Cache per station-year
                    os.makedirs(PATHS['cache_era5'], exist_ok=True)
                    result.to_parquet(cache_path, engine='pyarrow', index=False)
                    all_era5_dfs.append(result)

            ds.close()
            clean_memory()
            print(f"  {year}: completed ({len(stations)} stations)")

        # Merge all ERA5 data back into main DataFrame
        if all_era5_dfs:
            df_era5 = pd.concat(all_era5_dfs, ignore_index=True)
            del all_era5_dfs
            clean_memory()
            
            # ENSURE SCHEMA BEFORE MERGE (Critical Fix)
            from src.utils import enforce_schema
            df_era5 = enforce_schema(df_era5, source_name="ERA5_CONCAT")
            df = enforce_schema(df, source_name="MAIN_PIPELINE")

            # Merge on station + timestamp
            era5_merge_cols = ['station', 'timestamp'] + \
                [c for c in df_era5.columns
                 if c not in ('station', 'timestamp')]

            # Drop existing ERA5 columns if re-merging
            existing = [c for c in ERA5_VARS if c in df.columns]
            if existing:
                df.drop(columns=existing, inplace=True)

            df = df.merge(df_era5[era5_merge_cols],
                          on=['station', 'timestamp'], how='left')
            del df_era5
            clean_memory()

            # Validation
            print("\n[ERA5 VALIDATION]")
            for var in ERA5_VARS:
                if var in df.columns:
                    n_nan = df[var].isna().sum()
                    pct = 100 * n_nan / len(df)
                    print(f"  {var:15}: NaN={n_nan:7} ({pct:5.1f}%), "
                          f"range=[{df[var].min():8.2f}, {df[var].max():8.2f}]")
        else:
            print("  WARNING: No ERA5 data was processed!")
            for var in ERA5_VARS:
                if var not in df.columns:
                    df[var] = np.float32(np.nan)

    return df
