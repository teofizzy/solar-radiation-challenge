"""
Shared utilities: seeding, memory optimization, caching, and validation helpers.
"""

import os
import gc
import time
import contextlib
import numpy as np
import pandas as pd

from src.config import DTYPE, DTYPE_STR


# ------------------------------------------------------------------
# Memory optimization
# ------------------------------------------------------------------
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggressively downcast DataFrame columns to minimize memory footprint.
    Enforces float32 ceiling (never float64) per dtype policy.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object' or str(col_type) == 'category':
            continue

        if str(col_type).startswith('float'):
            df[col] = df[col].astype(np.float32)

        elif str(col_type).startswith('int'):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"  Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({pct:.1f}% reduction)")
    return df


# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------
def load_or_compute(cache_path: str, compute_fn, force_recompute: bool = False,
                    verbose: bool = True):
    """
    Load from parquet cache if it exists; otherwise run compute_fn and save.

    Parameters
    ----------
    cache_path : str
        Path to parquet file.
    compute_fn : callable
        Zero-argument function that returns a DataFrame.
    force_recompute : bool
        If True, ignore cache and recompute.

    Returns
    -------
    pd.DataFrame
    """
    if os.path.exists(cache_path) and not force_recompute:
        if verbose:
            print(f"  [CACHE HIT] {os.path.basename(cache_path)}")
        return pd.read_parquet(cache_path)

    if verbose:
        print(f"  [COMPUTING] {os.path.basename(cache_path)}...")

    df = compute_fn()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, engine='pyarrow', index=True)

    if verbose:
        print(f"  [CACHED] {os.path.basename(cache_path)} ({len(df)} rows)")
    return df


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------
def validate_no_nan(df: pd.DataFrame, name: str, columns: list = None):
    """Assert no NaN values in specified columns (or all columns)."""
    cols = columns if columns else df.columns.tolist()
    for col in cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"  WARNING: {name}.{col} has {n_nan} NaN values "
                      f"({100*n_nan/len(df):.2f}%)")


def validate_range(series: pd.Series, name: str, low: float, high: float):
    """Assert values are within [low, high] range."""
    below = (series < low).sum()
    above = (series > high).sum()
    if below > 0:
        print(f"  WARNING: {name} has {below} values below {low} "
              f"(min={series.min():.4f})")
    if above > 0:
        print(f"  WARNING: {name} has {above} values above {high} "
              f"(max={series.max():.4f})")


def print_df_info(df: pd.DataFrame, name: str):
    """Print shape, memory, and NaN summary of a DataFrame."""
    mem = df.memory_usage(deep=True).sum() / 1024**2
    n_nan = df.isna().sum().sum()
    print(f"  {name}: shape={df.shape}, memory={mem:.1f}MB, total_NaN={n_nan}")


# ------------------------------------------------------------------
# Timer context manager
# ------------------------------------------------------------------
@contextlib.contextmanager
def timer(name: str = ""):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{name}] completed in {elapsed:.1f}s")


# ------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------
def get_device():
    """Return the best available torch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except ImportError:
        return None


def clean_memory():
    """Force garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'timestamp' is a column and correctly typed across all pipeline stages.
    Enforces canonical naming and avoids index-vs-column confusion.
    """
    df = df.copy()
    
    # 1. Handle timestamp in index
    if df.index.name == "timestamp" or "timestamp" not in df.columns:
        df = df.reset_index()
        
    # 2. Ensure column exists (might have been named 'time' or 'valid_time')
    if "timestamp" not in df.columns:
        # Check for synonyms
        synonyms = ["time", "valid_time", "Datetime", "date"]
        for syn in synonyms:
            if syn in df.columns:
                df = df.rename(columns={syn: "timestamp"})
                break
                
    if "timestamp" not in df.columns:
        raise KeyError("Could not find 'timestamp' or suitable synonym in DataFrame columns.")

    # 3. Canonical conversion
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    
    # 4. Standardize station names/types if present
    if "station" in df.columns:
        df["station"] = df["station"].astype(str)
        
    return df
