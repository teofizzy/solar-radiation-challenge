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
        log_clearsky_ghi, log_precipitation, log_wind_speed,
        t2m_celsius, d2m_celsius,
        t_lapse_corr, p_hyps_corr, csghi_terrain_corr

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
        'hour_sin', 'hour_cos', 'hour_12_sin', 'hour_12_cos', 
        'hour_6_sin', 'hour_6_cos', 'hour_3_sin', 'hour_3_cos',
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'log_clearsky_ghi',
        'log_precipitation', 'log_wind_speed',
        't2m_celsius', 'd2m_celsius', 'days_since_start',
        't_lapse_corr', 'p_hyps_corr', 'csghi_terrain_corr'
    ]

    recompute = force_recompute
    if os.path.exists(cache_path) and not recompute:
        try:
            df_phys = pd.read_parquet(cache_path)
            missing = [c for c in physics_cols if c not in df_phys.columns]
            if missing:
                print(f"[FEATURE_PHYSICS] Cache missing columns {missing}. Recomputing...")
                recompute = True
            else:
                print("[FEATURE_PHYSICS] Loading from cache...")
                existing = [c for c in physics_cols if c in df.columns]
                if existing:
                    df.drop(columns=existing, inplace=True)
                df = df.merge(df_phys[physics_cols], left_index=True,
                              right_index=True, how='left')
                return df
        except Exception as e:
            print(f"[FEATURE_PHYSICS] Cache error: {e}. Recomputing...")
            recompute = True

    with timer("FEATURE_PHYSICS"):
        print("[FEATURE_PHYSICS] Computing physics-derived features...")

        # ----------------------------------------------------------
        # 0. Global Sensor Drift (days since start)
        # ----------------------------------------------------------
        min_date = df['timestamp'].min()
        df['days_since_start'] = (df['timestamp'] - min_date).dt.total_seconds() / (24.0 * 3600.0)
        df['days_since_start'] = df['days_since_start'].astype(DTYPE)

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
        # 3b. Topographic Corrections (Temperature & Pressure)
        # ----------------------------------------------------------
        z_station = df['dem'].values if 'dem' in df.columns else df['elevation'].values
        # ERA5 elevation = geopotential (z) / g
        z_era5 = (df['z'].values / 9.80665) if 'z' in df.columns else z_station
        dz = z_station - z_era5

        # Lapse rate correction (standard 0.0065 K/m)
        df['t_lapse_corr'] = (df['t2m_celsius'] - 0.0065 * dz).astype(DTYPE)

        # Hypsometric pressure correction
        if 'sp' in df.columns:
            # P_station = P_era5 * exp(-g * dz / (R * T_mean))
            t_mean = (df['t2m'].values + (df['t_lapse_corr'].values + 273.15)) / 2.0
            p_corr = df['sp'].values * np.exp(-9.80665 * dz / (287.05 * t_mean))
            df['p_hyps_corr'] = p_corr.astype(DTYPE)
        else:
            df['p_hyps_corr'] = np.float32(101325.0)

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
        
        # Monthly encoding pruned as DOY is more continuous

        doy = df['dayofyear'].values.astype(np.float32)
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25).astype(DTYPE)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25).astype(DTYPE)

        # ----------------------------------------------------------
        # 6b. Terrain-Adjusted Clear Sky GHI
        # ----------------------------------------------------------
        if all(c in df.columns for c in ['slope', 'aspect_sin', 'aspect_cos', 'solar_zenith', 'solar_azimuth']):
            # cos(theta_i) = cos(z)cos(slope) + sin(z)sin(slope)cos(azimuth - aspect)
            sz = np.radians(df['solar_zenith'].values)
            slope = np.radians(df['slope'].values)
            azimuth = np.radians(df['solar_azimuth'].values)
            aspect = np.arctan2(df['aspect_sin'].values, df['aspect_cos'].values)
            
            cos_theta_i = (np.cos(sz) * np.cos(slope) + 
                           np.sin(sz) * np.sin(slope) * np.cos(azimuth - aspect))
            
            # Correction factor: cos(theta_i) / cos(theta_z)
            # Clip zenith to 85 to avoid division by zero
            cos_sz_clip = np.cos(np.clip(sz, 0, np.radians(85)))
            terrain_factor = np.clip(cos_theta_i / cos_sz_clip, 0.5, 2.0)
            
            df['csghi_terrain_corr'] = (df['clear_sky_ghi'].values * terrain_factor).astype(DTYPE)
        else:
            df['csghi_terrain_corr'] = df['clear_sky_ghi'].values

        # ----------------------------------------------------------
        # 7. Log transforms for skewed variables
        # ----------------------------------------------------------
        df['log_clearsky_ghi'] = np.log1p(df['csghi_terrain_corr'].values).astype(DTYPE)

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
