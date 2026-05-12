"""
Astronomical feature engineering using pvlib.
Computes clear-sky GHI (Ineichen model), solar position, and the clearness index (kt) target.

This is the MOST CRITICAL feature module -- kt = radiation / clear_sky_ghi is the learning target.
Errors here (timezone bugs, wrong clear-sky model) propagate to every downstream module.
"""

import os
import numpy as np
import pandas as pd
import pvlib

from src.config import PATHS, HPARAMS, DTYPE, ensure_dirs
from src.utils import timer, validate_range, print_df_info


def compute_astro_features(df: pd.DataFrame,
                           force_recompute: bool = False) -> pd.DataFrame:
    """
    Compute astronomical features and clearness index target for all stations.

    Adds columns:
        solar_zenith, solar_azimuth, cos_zenith, clear_sky_ghi,
        is_night, kt (clearness index target, NaN for test rows)

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame from data_loader with timestamp, station,
        latitude, longitude, elevation, radiation columns.
    force_recompute : bool
        If True, recompute even if cache exists.

    Returns
    -------
    pd.DataFrame with astronomical features added.
    """
    ensure_dirs()
    cache_path = os.path.join(PATHS['cache_astro'], 'astro_features.parquet')

    recompute = force_recompute
    if os.path.exists(cache_path) and not recompute:
        try:
            df_astro = pd.read_parquet(cache_path)
            astro_cols = ['solar_zenith', 'solar_azimuth', 'cos_zenith',
                          'clear_sky_ghi', 'is_night', 'kt']
            missing = [c for c in astro_cols if c not in df_astro.columns]
            if missing:
                print(f"[FEATURE_ASTRO] Cache missing columns {missing}. Recomputing...")
                recompute = True
            else:
                print("[FEATURE_ASTRO] Loading from cache...")
                existing = [c for c in astro_cols if c in df.columns]
                if existing:
                    df.drop(columns=existing, inplace=True)
                df = df.merge(df_astro[astro_cols], left_index=True, right_index=True,
                              how='left')
                return df
        except Exception as e:
            print(f"[FEATURE_ASTRO] Cache error: {e}. Recomputing...")
            recompute = True

    with timer("FEATURE_ASTRO"):
        print("[FEATURE_ASTRO] Computing solar position and clear-sky GHI...")

        ghi_list = []
        zenith_list = []
        azimuth_list = []

        stations = sorted(df['station'].unique())
        for i, station_id in enumerate(stations):
            mask = df['station'] == station_id
            group = df.loc[mask]
            loc_data = group.iloc[0]

            site = pvlib.location.Location(
                latitude=loc_data['latitude'],
                longitude=loc_data['longitude'],
                tz='UTC',
                altitude=loc_data['elevation']
            )

            times = pd.DatetimeIndex(group['timestamp'])

            # Solar position
            solar_pos = site.get_solarposition(times)

            # Clear-sky GHI using Ineichen model (uses static Linke Turbidity)
            clear_sky = site.get_clearsky(times, model='ineichen')

            ghi = clear_sky['ghi'].values.copy().astype(DTYPE)
            zenith = solar_pos['zenith'].values.astype(DTYPE)
            azimuth = solar_pos['azimuth'].values.astype(DTYPE)

            # Enforce exact nighttime = 0
            night_mask = zenith > HPARAMS['night_zenith_threshold']
            ghi[night_mask] = 0.0

            ghi_list.append(pd.Series(ghi, index=group.index, dtype=DTYPE))
            zenith_list.append(pd.Series(zenith, index=group.index, dtype=DTYPE))
            azimuth_list.append(pd.Series(azimuth, index=group.index, dtype=DTYPE))

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(stations)} stations")

        print(f"  Processed {len(stations)}/{len(stations)} stations")

        # Assign features
        df['clear_sky_ghi'] = pd.concat(ghi_list).sort_index()
        df['solar_zenith'] = pd.concat(zenith_list).sort_index()
        df['solar_azimuth'] = pd.concat(azimuth_list).sort_index()
        df['cos_zenith'] = np.cos(np.radians(df['solar_zenith'])).astype(DTYPE)
        df['is_night'] = (df['solar_zenith'] > HPARAMS['night_zenith_threshold']).astype(np.uint8)

        # ----------------------------------------------------------
        # Compute clearness index (kt) -- the LEARNING TARGET
        # Only for rows with measured radiation (training data)
        # ----------------------------------------------------------
        safe_csghi = np.maximum(df['clear_sky_ghi'].values,
                                HPARAMS['clearsky_min_denom'])
        df['kt'] = np.float32(np.nan)  # Default NaN

        has_radiation = df['radiation'].notna()
        kt_values = df.loc[has_radiation, 'radiation'].values / safe_csghi[has_radiation.values]

        # Clamp kt to physical bounds
        kt_values = np.clip(kt_values, 0.0, HPARAMS['kt_max']).astype(DTYPE)

        # Force kt = 0 at night
        night_train = df.loc[has_radiation, 'is_night'].values.astype(bool)
        kt_values[night_train] = 0.0

        df.loc[has_radiation, 'kt'] = kt_values

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        print("\n[ASTRO VALIDATION]")
        print(f"  clear_sky_ghi range: [{df['clear_sky_ghi'].min():.1f}, "
              f"{df['clear_sky_ghi'].max():.1f}]")
        print(f"  solar_zenith range:  [{df['solar_zenith'].min():.1f}, "
              f"{df['solar_zenith'].max():.1f}]")
        print(f"  Night fraction:      {df['is_night'].mean():.3f}")

        # kt validation (train only)
        kt_valid = df.loc[has_radiation, 'kt']
        kt_day = kt_valid[df.loc[has_radiation, 'is_night'] == 0]
        if len(kt_day) > 0:
            print(f"  kt (daytime, train): mean={kt_day.mean():.3f}, "
                  f"std={kt_day.std():.3f}, "
                  f"[{kt_day.min():.3f}, {kt_day.max():.3f}]")
            validate_range(kt_day, 'kt_daytime', 0.0, HPARAMS['kt_max'])

        # Verify clear_sky_ghi non-negative
        assert (df['clear_sky_ghi'] >= 0).all(), \
            "clear_sky_ghi has negative values!"

        # Verify nighttime clear_sky_ghi == 0
        night_nonzero = (df.loc[df['is_night'] == 1, 'clear_sky_ghi'] > 0).sum()
        if night_nonzero > 0:
            print(f"  WARNING: {night_nonzero} nighttime rows have nonzero clear_sky_ghi")

        # ----------------------------------------------------------
        # Cache
        # ----------------------------------------------------------
        astro_cols = ['solar_zenith', 'solar_azimuth', 'cos_zenith',
                      'clear_sky_ghi', 'is_night', 'kt']
        df[astro_cols].to_parquet(cache_path, engine='pyarrow')
        print(f"  Cached astro features to {cache_path}")

    return df
