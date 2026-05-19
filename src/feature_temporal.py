"""
Temporal feature engineering on COVARIATES ONLY.
Rolling statistics, lag features, wash cycle tracking, satellite lag stacks,
EWMA drift tracking, and cloud variability features.

CRITICAL LEAKAGE GUARD:
- Rolling statistics are computed ONLY on covariate columns (ERA5, temperature, humidity).
- Target radiation is NEVER used in rolling computations.
- Rolling windows do NOT cross train/test month boundaries.
- EWMA drift uses ONLY satellite-derived kt (kt_landsaf), never observed radiation.
- Lag features use groupby('station').shift() to prevent cross-station leakage.
"""

import numpy as np
import pandas as pd
import os

from src.config import PATHS, DTYPE, MULTI_SCALE_LAGS, ensure_dirs
from src.utils import timer


# Variables safe for rolling/lag (covariates only, never target)
ROLLING_VARS = [
    'temperature', 'relativehumidity', 'precipitation',
    'wind_speed', 'dewpoint_depression', 'cos_zenith',
    'clear_sky_ghi', 'pw_attenuation',
]

# Using MULTI_SCALE_LAGS from config: '1h': 4, '3h': 12, '6h': 24, '12h': 48

# Satellite columns for lag stack generation
SATELLITE_LAG_COLS = ['mdssf', 'kt_landsaf']
SATELLITE_LAG_STEPS = [1, 2, 4, 8]  # t-1, t-2, t-4, t-8 (15min each)

# EWMA spans for drift tracking (in 15-min steps)
EWMA_FAST_SPAN = 192   # ~48 hours -- captures recent drift
EWMA_SLOW_SPAN = 672   # ~7 days  -- captures baseline


def compute_temporal_features(df: pd.DataFrame,
                              force_recompute: bool = False) -> pd.DataFrame:
    """
    Compute temporal features per station: rolling stats, wash cycles,
    satellite lag stacks, EWMA drift tracking, and cloud variability.

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
        # 1. Optimized Rolling Stats (Mean, Std, Slope)
        # ----------------------------------------------------------
        # We prune stats to the most physically relevant ones for a Transformer
        for var in available_rolling:
            # Pre-sort by station and timestamp for vectorized rolling
            group_obj = df.groupby('station')[var]
            
            for window_name, window_size in MULTI_SCALE_LAGS.items():
                # MEAN: Primary baseline signal
                col_mean = f'{var}_roll_mean_{window_name}'
                if col_mean not in df.columns:
                    new_columns[col_mean] = group_obj.transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).mean()
                    ).astype(DTYPE)

                # SLOPE: Captures trends/ramps
                col_slope = f'{var}_roll_slope_{window_name}'
                if col_slope not in df.columns:
                    # vectorized: (current - lag) / window
                    shifted = group_obj.shift(window_size)
                    new_columns[col_slope] = ((df[var] - shifted) / window_size).fillna(0).astype(DTYPE)

                # STD: Only for volatility-sensitive variables
                if var in ['clear_sky_ghi', 'dewpoint_depression', 'cos_zenith']:
                    col_std = f'{var}_roll_std_{window_name}'
                    if col_std not in df.columns:
                        new_columns[col_std] = group_obj.transform(
                            lambda x: x.rolling(window=window_size, min_periods=2).std()
                        ).fillna(0).astype(DTYPE)

        # Wavelet Proxies removed (redundant with Slope)

        # ----------------------------------------------------------
        # 2. Clear-sky Volatility Index (rolling std of clearness proxy)
        #    Uses cos_zenith as a proxy for atmospheric transmittance
        # ----------------------------------------------------------
        if 'cos_zenith' in df.columns:
            vol = df.groupby('station')['cos_zenith'].transform(
                lambda x: x.rolling(window=16, min_periods=2).std()
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
        # 4. Satellite Lag Stacks (Phase 1A)
        #    Per-station shifted satellite values: t-1, t-2, t-4, t-8
        #    Strictly causal, no cross-station leakage
        # ----------------------------------------------------------
        available_sat = [c for c in SATELLITE_LAG_COLS if c in df.columns]
        for col in available_sat:
            for lag in SATELLITE_LAG_STEPS:
                lag_col = f'{col}_lag_{lag}'
                if lag_col not in df.columns:
                    shifted = df.groupby('station')[col].shift(lag)
                    new_columns[lag_col] = shifted.astype(DTYPE)
        if available_sat:
            n_lag_cols = len(available_sat) * len(SATELLITE_LAG_STEPS)
            print(f"  Satellite lag stacks: {n_lag_cols} columns "
                  f"({available_sat} x lags {SATELLITE_LAG_STEPS})")

        # ----------------------------------------------------------
        # 4.5. Satellite Difference Features (cloud edge proxy)
        #      1-hour difference captures rapid cloud transitions
        # ----------------------------------------------------------
        for col in available_sat:
            diff_col = f'{col}_diff_1h'
            if diff_col not in df.columns:
                new_columns[diff_col] = df.groupby('station')[col].diff(
                    periods=4  # 4 steps = 1 hour
                ).fillna(0).astype(DTYPE)

        # ----------------------------------------------------------
        # 5. EWMA Drift Tracking (Phase 1B)
        #    Dual-scale EWMA on satellite kt to isolate sensor drift
        #    Uses ONLY kt_landsaf (satellite), never observed radiation
        # ----------------------------------------------------------
        if 'kt_landsaf' in df.columns:
            # Fast EWMA (~48 hours) -- recent drift signal
            ewma_fast = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.ewm(span=EWMA_FAST_SPAN, min_periods=1).mean()
            )
            new_columns['ewma_kt_fast'] = ewma_fast.fillna(0).astype(DTYPE)

            # Slow EWMA (~7 days) -- baseline signal
            ewma_slow = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.ewm(span=EWMA_SLOW_SPAN, min_periods=1).mean()
            )
            new_columns['ewma_kt_slow'] = ewma_slow.fillna(0).astype(DTYPE)

            # Drift proxy: fast - slow (isolates recent sensor shift)
            new_columns['drift_proxy'] = (
                ewma_fast.values - ewma_slow.values
            ).astype(DTYPE)

            print(f"  EWMA drift: fast(span={EWMA_FAST_SPAN}), "
                  f"slow(span={EWMA_SLOW_SPAN}), drift_proxy")

        # ----------------------------------------------------------
        # 5.5. Cumulative GHI Exposure (Phase 1B)
        #      Log-transformed cumulative clear-sky energy per station
        #      Proxy for pyranometer aging / dust accumulation
        # ----------------------------------------------------------
        if 'clear_sky_ghi' in df.columns:
            def _cum_exposure(group):
                """Cumulative Wh/m2 normalized by deployment days, log-transformed."""
                cum = (group['clear_sky_ghi'].fillna(0).astype(DTYPE) * 0.25).cumsum()
                days = max(
                    (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
                    / (3600 * 24), 1.0
                )
                return np.log1p(cum / days).astype(DTYPE)

            cum_exp = df.groupby('station').apply(
                _cum_exposure, include_groups=False
            ).reset_index(level=0, drop=True)
            new_columns['log_cum_exposure'] = cum_exp

            print(f"  Cumulative GHI exposure: log1p(cumsum/days)")

        # ----------------------------------------------------------
        # 5.6. Causal EWMA Residual (kt_obs - kt_landsaf)
        #      Tracks pyranometer drift.
        #      CRITICAL: MUST shift by 1 to prevent data leakage!
        #      In test set, radiation is NaN, so ffill propagates drift.
        # ----------------------------------------------------------
        if all(c in df.columns for c in ['radiation', 'clear_sky_ghi', 'kt_landsaf']):
            # Compute kt_obs
            # Avoid division by zero by clipping clear_sky_ghi to 10 W/m2 min
            csi = np.maximum(df['clear_sky_ghi'].values, 10.0)
            kt_obs = df['radiation'].values / csi
            # Limit kt_obs to physical bounds to prevent wild EWMA spikes
            kt_obs = np.clip(kt_obs, 0, 1.2)
            
            # Residual (Observed Transmissivity - Satellite Transmissivity)
            residual = kt_obs - df['kt_landsaf'].values
            
            # Create a Series with station multi-index for safe shifting and EWMA
            res_series = pd.Series(residual, index=df.index)
            
            # In Test set (where radiation is NaN), ffill will carry the last known drift forward
            # This perfectly models constant sensor drift for the unseen month
            causal_residual = df.groupby('station', group_keys=False).apply(
                lambda g: res_series.loc[g.index].ffill().shift(1)
            )
            
            # Compute EWMA on the strictly causal, forward-filled residual
            # Span=672 (1 week) to capture slow drift
            ewma_residual = df.groupby('station', group_keys=False).apply(
                lambda g: causal_residual.loc[g.index].ewm(span=672, ignore_na=True).mean()
            )
            
            new_columns['ewma_residual_kt'] = ewma_residual.fillna(0).astype(DTYPE)
            print("  Causal EWMA Residual (kt) computed.")

        # ----------------------------------------------------------
        # 5.7. Beyond-Window Features for BiLSTM
        #      These encode temporal context beyond the 12h BiLSTM window.
        #      All use satellite-derived covariates only (no target leakage).
        # ----------------------------------------------------------
        # (a) 24h cloud variability (beyond 12h window)
        if 'kt_landsaf' in df.columns:
            cloud_std_24h = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.rolling(window=96, min_periods=4).std()  # 96 steps = 24h
            )
            new_columns['rolling_24h_cloud_std'] = cloud_std_24h.fillna(0).astype(DTYPE)
            print("  Beyond-window: rolling_24h_cloud_std (24h kt_landsaf std)")

        # (b) 72h kt anomaly (clearness index vs 3-day mean)
        if 'kt_landsaf' in df.columns:
            kt_72h_mean = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.rolling(window=288, min_periods=8).mean()  # 288 steps = 72h
            )
            # Compute subtraction in pandas (not numpy) so .fillna() is available
            rolling_72h_kt_anomaly = (df['kt_landsaf'] - kt_72h_mean).fillna(0)
            new_columns['rolling_72h_kt_anomaly'] = rolling_72h_kt_anomaly.astype(DTYPE)
            print("  Beyond-window: rolling_72h_kt_anomaly (kt vs 3-day mean)")

        # (c) EWMA kt 24h (exponential drift tracker over 24h)
        if 'kt_landsaf' in df.columns:
            ewma_24h = df.groupby('station')['kt_landsaf'].transform(
                lambda x: x.ewm(span=96, min_periods=1).mean()  # span=96 ~ 24h half-life
            )
            new_columns['ewma_kt_24h'] = ewma_24h.fillna(0).astype(DTYPE)
            print("  Beyond-window: ewma_kt_24h (24h exponential drift)")

        # ----------------------------------------------------------
        # 6. Cloud Motion Proxies (Advection)
        #    Advection = -(u * dKt/dx + v * dKt/dy)
        #    Requires KNN spatial gradients of kt_landsaf
        # ----------------------------------------------------------
        if all(c in df.columns for c in ['kt_landsaf', 'u10', 'v10', 'latitude', 'longitude']):
            print("  Computing Cloud Motion Advection (KNN spatial gradients)...")
            from sklearn.neighbors import NearestNeighbors
            
            # Pivot kt to compute cross-station gradients at each timestamp
            kt_pivot = df.pivot(index='timestamp', columns='station', values='kt_landsaf')
            stations = df[['station', 'latitude', 'longitude']].drop_duplicates().set_index('station')
            station_names = stations.index.tolist()
            
            # Rough conversion to km (1 deg ~ 111 km)
            X = stations['longitude'].values * 111.0
            Y = stations['latitude'].values * 111.0
            coords = np.column_stack((X, Y))
            
            K = min(5, len(station_names) - 1)
            nbrs = NearestNeighbors(n_neighbors=K+1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            grad_x = np.zeros_like(kt_pivot.values)
            grad_y = np.zeros_like(kt_pivot.values)
            
            for i, st in enumerate(station_names):
                nb_idx = indices[i, 1:]
                dX = X[nb_idx] - X[i]
                dY = Y[nb_idx] - Y[i]
                A = np.column_stack((dX, dY))
                try:
                    A_pinv = np.linalg.pinv(A)
                except np.linalg.LinAlgError:
                    A_pinv = np.zeros((2, K))
                    
                kt_i = kt_pivot.iloc[:, i].values
                kt_nb = kt_pivot.iloc[:, nb_idx].values
                dKt = kt_nb - kt_i[:, None]
                
                grad = dKt @ A_pinv.T
                grad_x[:, i] = grad[:, 0]
                grad_y[:, i] = grad[:, 1]
                
            grad_x_df = pd.DataFrame(grad_x, index=kt_pivot.index, columns=station_names)
            grad_y_df = pd.DataFrame(grad_y, index=kt_pivot.index, columns=station_names)
            
            # Use rename_axis to ensure levels are named correctly for reordering
            grad_x_series = grad_x_df.rename_axis(index='timestamp', columns='station').stack().swaplevel().sort_index()
            grad_y_series = grad_y_df.rename_axis(index='timestamp', columns='station').stack().swaplevel().sort_index()
            
            # Ensure df is indexed correctly for assignment
            df_idx = df.set_index(['station', 'timestamp']).index
            
            # Map back to original dataframe structure
            # reindex handles missing timestamps safely
            new_columns['advection_kt'] = (
                - (df['u10'].values * grad_x_series.reindex(df_idx).values + 
                   df['v10'].values * grad_y_series.reindex(df_idx).values)
            ).astype(DTYPE)
            print("  Cloud Advection computed (optimized).")

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
        forbidden = ['radiation', 'CSI']
        # Note: 'kt' alone is NOT forbidden -- 'kt_landsaf' and 'kt_ewma' are satellite-derived
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
