"""
TAHMO CSV data ingestion with aggressive dtype optimization.
Loads Train.csv and Test.csv, merges station metadata, and caches per-station.
"""

import os
import numpy as np
import pandas as pd

from src.config import PATHS, DTYPE, ensure_dirs
from src.utils import reduce_mem_usage, print_df_info, timer


# Column dtype specifications for efficient loading
TRAIN_DTYPES = {
    'precipitation (mm)': np.float32,
    'radiation (W/m2)': np.float32,
    'relativehumidity (-)': np.float32,
    'temperature (degrees Celsius)': np.float32,
    'elevation': np.float32,
    'latitude': np.float64,   # Keep precision for pvlib
    'longitude': np.float64,  # Keep precision for pvlib
}

TEST_DTYPES = {
    'precipitation (mm)': np.float32,
    'relativehumidity (-)': np.float32,
    'temperature (degrees Celsius)': np.float32,
    'elevation': np.float32,
    'latitude': np.float64,
    'longitude': np.float64,
}


def load_raw_data(force_recompute: bool = False) -> pd.DataFrame:
    """
    Load and merge Train + Test CSVs into a single DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        ID, timestamp, precipitation, radiation (NaN for test),
        relativehumidity, temperature, station, latitude, longitude,
        elevation, is_test, station_idx
    """
    ensure_dirs()
    cache_path = os.path.join(PATHS['cache_raw'], 'combined.parquet')

    if os.path.exists(cache_path) and not force_recompute:
        print("[DATA_LOADER] Loading from cache...")
        df = pd.read_parquet(cache_path)
        print_df_info(df, "combined")
        return df

    with timer("DATA_LOADER"):
        # -- Load Train --
        print("[DATA_LOADER] Loading Train.csv...")
        df_train = pd.read_csv(PATHS['train'], dtype=TRAIN_DTYPES, parse_dates=['timestamp'])
        df_train['is_test'] = np.uint8(0)
        print(f"  Train: {df_train.shape}")

        # -- Load Test --
        print("[DATA_LOADER] Loading Test.csv...")
        df_test = pd.read_csv(PATHS['test'], dtype=TEST_DTYPES, parse_dates=['timestamp'])
        df_test['is_test'] = np.uint8(1)
        df_test['radiation (W/m2)'] = np.float32(np.nan)
        print(f"  Test: {df_test.shape}")

        # -- Combine --
        df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
        del df_train, df_test

        # -- Rename columns to clean names --
        rename_map = {
            'precipitation (mm)': 'precipitation',
            'radiation (W/m2)': 'radiation',
            'relativehumidity (-)': 'relativehumidity',
            'temperature (degrees Celsius)': 'temperature',
        }
        df.rename(columns=rename_map, inplace=True)

        # -- Drop columns we do not need (already in station_meta) --
        drop_cols = ['station_name', 'country', 'installation_height']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # -- Create integer station index --
        stations_sorted = sorted(df['station'].unique())
        station_to_idx = {s: i for i, s in enumerate(stations_sorted)}
        df['station_idx'] = df['station'].map(station_to_idx).astype(np.int16)

        # -- Sort for temporal consistency --
        df.sort_values(['station', 'timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # -- Extract time components --
        df['year'] = df['timestamp'].dt.year.astype(np.int16)
        df['month'] = df['timestamp'].dt.month.astype(np.uint8)
        df['day'] = df['timestamp'].dt.day.astype(np.uint8)
        df['hour'] = df['timestamp'].dt.hour.astype(np.uint8)
        df['minute'] = df['timestamp'].dt.minute.astype(np.uint8)
        df['dayofyear'] = df['timestamp'].dt.dayofyear.astype(np.int16)

        # -- Memory optimization --
        df = reduce_mem_usage(df, verbose=True)

        # -- Validation --
        print("\n[VALIDATION]")
        print(f"  Stations: {df['station'].nunique()}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Train rows: {(df['is_test'] == 0).sum():,}")
        print(f"  Test rows:  {(df['is_test'] == 1).sum():,}")

        # Check for duplicate timestamps per station
        dup_count = df.groupby('station').apply(
            lambda g: g['timestamp'].duplicated().sum()
        ).sum()
        if dup_count > 0:
            print(f"  WARNING: {dup_count} duplicate timestamps found!")
        else:
            print("  No duplicate timestamps (PASS)")

        print_df_info(df, "combined")

        # -- Cache --
        df.to_parquet(cache_path, engine='pyarrow', index=False)
        print(f"  Cached to {cache_path}")

    return df


def load_station_data(station_id: str) -> pd.DataFrame:
    """Load cached data for a single station."""
    cache_path = os.path.join(PATHS['cache_raw'],
                              f'{station_id}.parquet')
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    # If per-station cache does not exist, load from combined and filter
    df = load_raw_data()
    return df[df['station'] == station_id].copy()


def get_station_list(df: pd.DataFrame = None) -> list:
    """Return sorted list of station IDs."""
    if df is not None:
        return sorted(df['station'].unique().tolist())
    df = load_raw_data()
    return sorted(df['station'].unique().tolist())


def get_station_to_idx(df: pd.DataFrame = None) -> dict:
    """Return mapping from station ID to integer index."""
    stations = get_station_list(df)
    return {s: i for i, s in enumerate(stations)}
