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


def fix_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize coordinate names to 'latitude' and 'longitude'.
    Some ERA5 datasets use 'lat'/'lon'.
    """
    rename_map = {}
    if 'lat' in ds.coords and 'latitude' not in ds.coords:
        rename_map['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords:
        rename_map['lon'] = 'longitude'
    
    if rename_map:
        print(f"  [FEATURE_ERA5] Renaming coordinates: {rename_map}")
        return ds.rename(rename_map)
    return ds


def extract_era5_for_station(ds, station_id: str, lat: float, lon: float,
                             target_timestamps: pd.Series) -> pd.DataFrame:
    """
    Extract ERA5 variables for a single station and PCHIP-interpolate to 15-min.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 dataset (single year or month).
    station_id : str
        Station identifier.
    lat, lon : float
        Station coordinates.
    target_timestamps : pd.Series
        15-minute timestamps to interpolate to.

    Returns
    -------
    pd.DataFrame with ERA5 variables at 15-minute cadence.
    """
    # 1. Coordinate normalization (longitude wrap-around 0-360)
    lon = lon % 360
    
    # 2. Standardize naming
    ds = fix_coords(ds)
    
    # 3. Extract nearest grid point (bilinear interpolation)
    # Use method='linear' for smoothness, but nearest fallback if interp fails
    try:
        ds_point = ds.interp(latitude=lat, longitude=lon, method='linear')
    except Exception as e:
        print(f"  [FEATURE_ERA5] Interpolation failed for {station_id}: {e}. Falling back to nearest.")
        ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')

    # Select only the variables we need
    available_vars = [v for v in ERA5_VARS if v in ds_point.data_vars]
    if not available_vars:
        return pd.DataFrame()

    ds_point = ds_point[available_vars].compute()

    # Convert to DataFrame
    df_era = ds_point.to_dataframe()

    # Handle multi-index if expver dimension exists
    if isinstance(df_era.index, pd.MultiIndex):
        df_era = df_era.reset_index()
        if 'expver' in df_era.columns:
            # Take mean across expver (ERA5 vs ERA5T overlap)
            time_col = [c for c in df_era.columns if 'time' in c.lower()
                        or c == 'valid_time'][0]
            df_era = df_era.groupby(time_col)[available_vars].mean()
        else:
            time_col = [c for c in df_era.columns if 'time' in c.lower()
                        or c == 'valid_time'][0]
            df_era = df_era.set_index(time_col)

    # Ensure sorted, no duplicates
    df_era = df_era[~df_era.index.duplicated(keep='first')].sort_index()

    if len(df_era) < 2:
        return pd.DataFrame()

    # ERA5 timestamps as seconds since epoch
    era_times = df_era.index.astype(np.int64).values / 1e9
    # Target timestamps as seconds since epoch
    target_times = target_timestamps.astype(np.int64).values / 1e9

    # PCHIP interpolation per variable
    result = pd.DataFrame({'timestamp': target_timestamps})

    for var in available_vars:
        values = df_era[var].values

        # Fill small internal gaps via linear interpolation
        s = pd.Series(values)
        s = s.interpolate(method='linear', limit_direction='both')
        y_known = s.values

        if np.all(np.isnan(y_known)):
            result[var] = np.float32(np.nan)
            continue

        # PCHIP preserves monotonicity, avoids thermodynamic overshoots
        interp = PchipInterpolator(era_times, y_known, extrapolate=False)
        result[var] = interp(target_times).astype(DTYPE)

    return result


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
                for station_id in stations:
                    cache_path = os.path.join(
                        PATHS['cache_era5'], f'{station_id}_{year}.parquet')
                    all_era5_dfs.append(pd.read_parquet(cache_path))
                continue

            # Load ERA5 for this year
            print(f"  {year}: loading ERA5 NetCDF files...")
            nc_files = sorted(glob.glob(os.path.join(year_dir, '*.nc')))
            if not nc_files:
                print(f"  WARNING: No .nc files in {year_dir}")
                continue

            ds = xr.open_mfdataset(nc_files, combine='by_coords',
                                   chunks={'time': 100})
            
            # Diagnostic for coordinate ranges
            if 'latitude' in ds.coords or 'lat' in ds.coords:
                temp_ds = fix_coords(ds)
                lat_min, lat_max = float(temp_ds.latitude.min()), float(temp_ds.latitude.max())
                lon_min, lon_max = float(temp_ds.longitude.min()), float(temp_ds.longitude.max())
                print(f"  [FEATURE_ERA5] Bounds: Lat=[{lat_min:.2f}, {lat_max:.2f}], "
                      f"Lon=[{lon_min:.2f}, {lon_max:.2f}]")

            # Handle expver dimension
            if 'expver' in ds.dims:
                ds = ds.mean(dim='expver', skipna=True)

            # Process each station
            for station_id in stations:
                cache_path = os.path.join(
                    PATHS['cache_era5'], f'{station_id}_{year}.parquet')

                if os.path.exists(cache_path) and not force_recompute:
                    all_era5_dfs.append(pd.read_parquet(cache_path))
                    continue

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
                    print(f"  {var}: NaN={n_nan} ({pct:.1f}%), "
                          f"range=[{df[var].min():.2f}, {df[var].max():.2f}]")
        else:
            print("  WARNING: No ERA5 data was processed!")
            for var in ERA5_VARS:
                if var not in df.columns:
                    df[var] = np.float32(np.nan)

    return df
