"""
Physics-derived features using astronomical and ERA5 variables.
Radiative transfer proxies, atmospheric indicators, and cyclical time encoding.
"""

import numpy as np
import pandas as pd
import os

from src.config import PATHS, DTYPE, HPARAMS, ensure_dirs
from src.utils import timer, validate_range


def compute_physics_features(df: pd.DataFrame,
                             force_recompute: bool = False) -> pd.DataFrame:
    """
    Compute physics-derived features from astronomical + ERA5 variables.

    Requires: solar_zenith, cos_zenith, clear_sky_ghi (from feature_astro)
              u10, v10, d2m, t2m, sp, tco3, tcwv (from feature_era5)

    Adds columns:
        air_mass, wind_speed, wind_direction_sin, wind_direction_cos,
        dewpoint_depression, pw_attenuation, turbidity_proxy,
        hour_sin, hour_cos, month_sin, month_cos, doy_sin, doy_cos,
        log_clearsky_ghi, log_precipitation, log_wind_speed,
        t2m_celsius, d2m_celsius

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with astro and ERA5 features.
    force_recompute : bool
        If True, recompute from scratch.

    Returns
    -------
    pd.DataFrame with physics features added.
    """
    ensure_dirs()
    cache_path = os.path.join(PATHS['cache_physics'], 'physics_features.parquet')

    physics_cols = [
        'air_mass', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos',
        'dewpoint_depression', 'pw_attenuation', 'turbidity_proxy',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'doy_sin', 'doy_cos', 'log_clearsky_ghi',
        'log_precipitation', 'log_wind_speed',
        't2m_celsius', 'd2m_celsius',
    ]

    if os.path.exists(cache_path) and not force_recompute:
        print("[FEATURE_PHYSICS] Loading from cache...")
        df_phys = pd.read_parquet(cache_path)
        existing = [c for c in physics_cols if c in df.columns]
        if existing:
            df.drop(columns=existing, inplace=True)
        df = df.merge(df_phys[physics_cols], left_index=True,
                      right_index=True, how='left')
        return df

    with timer("FEATURE_PHYSICS"):
        print("[FEATURE_PHYSICS] Computing physics-derived features...")

        # ----------------------------------------------------------
        # 1. Air Mass (Kasten & Young, 1989)
        # AM = 1 / (cos(z) + 0.50572 * (96.07995 - z)^(-1.6364))
        # ----------------------------------------------------------
        zenith_rad = np.radians(df['solar_zenith'].values)
        zenith_deg = df['solar_zenith'].values

        # Clip zenith to avoid numerical instability at horizon
        z_clipped = np.clip(zenith_deg, 0, 89.99)
        denom = (np.cos(np.radians(z_clipped)) +
                 0.50572 * np.power(96.07995 - z_clipped, -1.6364))
        df['air_mass'] = np.where(
            zenith_deg < 90,
            (1.0 / np.maximum(denom, 1e-6)),
            np.float32(0.0)
        ).astype(DTYPE)

        # Clamp to reasonable range
        df['air_mass'] = np.clip(df['air_mass'].values, 0, 40).astype(DTYPE)

        # ----------------------------------------------------------
        # 2. Wind speed and direction from u10, v10
        #    Direction is cyclical (0==360), so decompose into sin/cos
        # ----------------------------------------------------------
        if 'u10' in df.columns and 'v10' in df.columns:
            u = df['u10'].values
            v = df['v10'].values
            df['wind_speed'] = np.sqrt(u**2 + v**2).astype(DTYPE)
            wind_dir_rad = np.arctan2(-u, -v)  # meteorological convention
            df['wind_direction_sin'] = np.sin(wind_dir_rad).astype(DTYPE)
            df['wind_direction_cos'] = np.cos(wind_dir_rad).astype(DTYPE)
        else:
            df['wind_speed'] = np.float32(0.0)
            df['wind_direction_sin'] = np.float32(0.0)
            df['wind_direction_cos'] = np.float32(0.0)

        # ----------------------------------------------------------
        # 2b. Log-transform wind speed (right-skewed)
        # ----------------------------------------------------------
        df['log_wind_speed'] = np.log1p(df['wind_speed'].values).astype(DTYPE)

        # ----------------------------------------------------------
        # 3. ERA5 Kelvin -> Celsius conversion
        #    ERA5 t2m/d2m are in K, TAHMO temperature is in C.
        #    Convert to same unit system before computing derived features.
        # ----------------------------------------------------------
        if 't2m' in df.columns:
            df['t2m_celsius'] = (df['t2m'].values - 273.15).astype(DTYPE)
        else:
            df['t2m_celsius'] = np.float32(0.0)

        if 'd2m' in df.columns:
            df['d2m_celsius'] = (df['d2m'].values - 273.15).astype(DTYPE)
        else:
            df['d2m_celsius'] = np.float32(0.0)

        # Dewpoint depression in Celsius (consistent with TAHMO temperature)
        if 't2m' in df.columns and 'd2m' in df.columns:
            df['dewpoint_depression'] = (df['t2m_celsius'] - df['d2m_celsius']).astype(DTYPE)
        else:
            df['dewpoint_depression'] = np.float32(0.0)

        # ----------------------------------------------------------
        # 4. Precipitable water attenuation
        # pw_att = exp(-0.1 * AM * tcwv)
        # ----------------------------------------------------------
        if 'tcwv' in df.columns:
            tcwv = df['tcwv'].fillna(0).values
            am = df['air_mass'].values
            df['pw_attenuation'] = np.exp(-0.1 * am * tcwv / 1000.0).astype(DTYPE)
        else:
            df['pw_attenuation'] = np.float32(1.0)

        # ----------------------------------------------------------
        # 5. Turbidity proxy (atmospheric opacity indicator)
        # ----------------------------------------------------------
        if all(c in df.columns for c in ['tcwv', 'tco3', 'sp']):
            tcwv = df['tcwv'].fillna(0).values
            tco3 = df['tco3'].fillna(0).values
            sp = np.maximum(df['sp'].fillna(101325).values, 1.0)  # Pa
            am = df['air_mass'].values

            df['turbidity_proxy'] = (
                (tcwv * np.maximum(am, 0.01) + tco3 * 10.0) / (sp / 1000.0)
            ).astype(DTYPE)
        else:
            df['turbidity_proxy'] = np.float32(0.0)

        # ----------------------------------------------------------
        # 6. Cyclical time encoding (sin/cos)
        # ----------------------------------------------------------
        hour_frac = df['hour'].values + df['minute'].values / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24.0).astype(DTYPE)
        df['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24.0).astype(DTYPE)

        month = df['month'].values.astype(np.float32)
        df['month_sin'] = np.sin(2 * np.pi * month / 12.0).astype(DTYPE)
        df['month_cos'] = np.cos(2 * np.pi * month / 12.0).astype(DTYPE)

        doy = df['dayofyear'].values.astype(np.float32)
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25).astype(DTYPE)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25).astype(DTYPE)

        # ----------------------------------------------------------
        # 7. Log transforms for skewed variables
        # ----------------------------------------------------------
        df['log_clearsky_ghi'] = np.log1p(df['clear_sky_ghi'].values).astype(DTYPE)

        # Log-transform precipitation (extreme right-skew, z-range ~62 -> ~5)
        if 'precipitation' in df.columns:
            df['log_precipitation'] = np.log1p(
                df['precipitation'].fillna(0).values
            ).astype(DTYPE)
        else:
            df['log_precipitation'] = np.float32(0.0)

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        print("\n[PHYSICS VALIDATION]")
        validate_range(df['air_mass'], 'air_mass', 0, 40)
        validate_range(df['wind_speed'], 'wind_speed', 0, 50)
        validate_range(df['pw_attenuation'], 'pw_attenuation', 0, 1.1)
        print(f"  air_mass (daytime mean): "
              f"{df.loc[df['is_night']==0, 'air_mass'].mean():.2f}")
        print(f"  wind_speed mean: {df['wind_speed'].mean():.2f} m/s")
        print(f"  dewpoint_depression mean: "
              f"{df['dewpoint_depression'].mean():.2f} K")

        # ----------------------------------------------------------
        # Cache
        # ----------------------------------------------------------
        df[physics_cols].to_parquet(cache_path, engine='pyarrow')
        print(f"  Cached physics features to {cache_path}")

    return df
