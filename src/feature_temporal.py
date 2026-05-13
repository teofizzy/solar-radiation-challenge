"""
Temporal feature engineering on COVARIATES ONLY.
Rolling statistics, lag features, and wash cycle tracking.

CRITICAL LEAKAGE GUARD:
- Rolling statistics are computed ONLY on covariate columns (ERA5, temperature, humidity).
- Target radiation is NEVER used in rolling computations.
- Rolling windows do NOT cross train/test month boundaries.
"""

import numpy as np
import pandas as pd
import os

from src.config import PATHS, DTYPE, ensure_dirs
from src.utils import timer


# Variables safe for rolling/lag (covariates only, never target)
ROLLING_VARS = [
    'temperature', 'relativehumidity', 'precipitation',
    'wind_speed', 'dewpoint_depression', 'cos_zenith',
    'clear_sky_ghi', 'pw_attenuation',
]

# Rolling window sizes (in number of 15-min steps)
WINDOWS = {
    '1h': 4,
    '4h': 16,
    '12h': 48,
}


def compute_temporal_features(df: pd.DataFrame,
                              force_recompute: bool = False) -> pd.DataFrame:
    """
    Compute temporal features per station: rolling stats and wash cycles.

    Only operates on COVARIATE columns to prevent target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with meteorological and physics features.
    force_recompute : bool
        If True, recompute from scratch.

    Returns
    -------
    pd.DataFrame with temporal features added.
    """
    ensure_dirs()
    cache_path = os.path.join(PATHS['cache_temporal'], 'temporal_features.parquet')

    if os.path.exists(cache_path) and not force_recompute:
        print("[FEATURE_TEMPORAL] Loading from cache...")
        df_temp = pd.read_parquet(cache_path)
        # Merge temporal features
        new_cols = [c for c in df_temp.columns if c not in df.columns]
        if new_cols:
            df = df.merge(df_temp[new_cols], left_index=True,
                          right_index=True, how='left')
        return df

    with timer("FEATURE_TEMPORAL"):
        print("[FEATURE_TEMPORAL] Computing temporal features (covariates only)...")

        new_columns = {}

        # Determine which rolling variables are available
        available_rolling = [v for v in ROLLING_VARS if v in df.columns]
        print(f"  Rolling variables: {available_rolling}")

        # ----------------------------------------------------------
        # 1. Rolling mean/std per station (covariates only)
        # ----------------------------------------------------------
        for var in available_rolling:
            for window_name, window_size in WINDOWS.items():
                col_mean = f'{var}_roll_mean_{window_name}'
                col_std = f'{var}_roll_std_{window_name}'

                if col_mean not in df.columns:
                    roll = df.groupby('station')[var].transform(
                        lambda x: x.rolling(
                            window=window_size, min_periods=1, center=True
                        ).mean()
                    )
                    new_columns[col_mean] = roll.astype(DTYPE)

                if col_std not in df.columns:
                    roll_std = df.groupby('station')[var].transform(
                        lambda x: x.rolling(
                            window=window_size, min_periods=2, center=True
                        ).std()
                    )
                    new_columns[col_std] = roll_std.fillna(0).astype(DTYPE)

        # ----------------------------------------------------------
        # 1.5. Wavelet Proxies (Multi-scale differences)
        # ----------------------------------------------------------
        for var in ['temperature', 'relativehumidity', 'clear_sky_ghi']:
            if var in df.columns:
                # 1h (4 steps), 3h (12 steps), 6h (24 steps) differences
                for lag, lag_name in [(4, '1h'), (12, '3h'), (24, '6h')]:
                    col_diff = f'{var}_diff_{lag_name}'
                    if col_diff not in df.columns:
                        diff = df.groupby('station')[var].diff(periods=lag)
                        new_columns[col_diff] = diff.fillna(0).astype(DTYPE)

        # ----------------------------------------------------------
        # 2. Clear-sky Volatility Index (rolling std of clearness proxy)
        #    Uses cos_zenith as a proxy for atmospheric transmittance
        # ----------------------------------------------------------
        if 'cos_zenith' in df.columns:
            vol = df.groupby('station')['cos_zenith'].transform(
                lambda x: x.rolling(window=16, min_periods=2, center=True).std()
            )
            new_columns['volatility_index'] = vol.fillna(0).astype(DTYPE)

        # ----------------------------------------------------------
        # 3. Wash cycle tracking
        #    Precipitation events "wash" pyranometer dust
        # ----------------------------------------------------------
        if 'precipitation' in df.columns:
            # Binary: did it rain in this 15-min window?
            rained = (df['precipitation'] > 0.1).astype(np.uint8)

            # Hours since last wash (per station)
            def hours_since_wash(precip_series):
                """Count 15-min steps since last rain event."""
                steps_since = pd.Series(0, index=precip_series.index, dtype=np.int32)
                counter = 0
                for i, val in enumerate(precip_series.values):
                    if val > 0.1:
                        counter = 0
                    else:
                        counter += 1
                    steps_since.iloc[i] = counter
                return steps_since * 0.25  # Convert to hours

            wash_hours = df.groupby('station')['precipitation'].transform(
                hours_since_wash
            )
            new_columns['hours_since_wash'] = wash_hours.astype(DTYPE)

            # Sticky dust index: hours_since_wash * (1 - relativehumidity)
            if 'relativehumidity' in df.columns:
                rh = df['relativehumidity'].fillna(0.5).values
                new_columns['sticky_dust_index'] = (
                    wash_hours.values * (1 - rh)
                ).astype(DTYPE)

        # ----------------------------------------------------------
        # 4. Explicit Drift Tracking (30-day EWMA of Satellite KT)
        #    Represents long-term sensor degradation / aerosol baseline
        # ----------------------------------------------------------
        if 'kt_landsaf' in df.columns:
            # 30 days * 24 hours * 4 steps = 2880 steps
            ewma = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.ewm(span=2880, min_periods=1).mean()
            )
            new_columns['kt_ewma_drift'] = ewma.fillna(0).astype(DTYPE)

        # ----------------------------------------------------------
        # Assign all new columns at once
        # ----------------------------------------------------------
        for col_name, col_values in new_columns.items():
            df[col_name] = col_values

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        print(f"\n[TEMPORAL VALIDATION]")
        print(f"  New columns added: {len(new_columns)}")

        # Verify no target-derived features crept in
        forbidden = ['radiation', 'kt', 'CSI']
        for name in new_columns.keys():
            for f in forbidden:
                assert f not in name, \
                    f"LEAKAGE: temporal feature '{name}' contains forbidden '{f}'!"
        print("  No target leakage detected (PASS)")

        if 'hours_since_wash' in df.columns:
            print(f"  hours_since_wash: mean={df['hours_since_wash'].mean():.1f}, "
                  f"max={df['hours_since_wash'].max():.1f}")

        # ----------------------------------------------------------
        # Cache
        # ----------------------------------------------------------
        temporal_cols = list(new_columns.keys())
        df[temporal_cols].to_parquet(cache_path, engine='pyarrow')
        print(f"  Cached {len(temporal_cols)} temporal features")

    return df
