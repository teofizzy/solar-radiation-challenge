"""
Post-processing pipeline for solar radiation predictions.

Implements:
  1. RTS (Rauch-Tung-Striebel) smoother for per-station bias correction
  2. Savitzky-Golay filter for residual high-frequency jitter removal

Correct ordering: Ensemble -> RTS Smoother -> Savitzky-Golay -> Physics Gate

State formulation for RTS:
  x_t = [kt_t, bias_t]
  F = [[1, 1], [0, 1]]   -- bias slowly accumulates
  H = [1, 0]              -- observe kt (bias not directly observed)
  Q = diag(q_kt, q_bias)  -- process noise
  R = [[r]]               -- observation noise

Q/R values from multi-AI consensus:
  Q = diag(0.0018, 0.00008)
  R = 0.012
"""

import numpy as np
from scipy.signal import savgol_filter

from src.config import HPARAMS


def rts_smoother_single_station(observations, q_kt=None, q_bias=None, r=None):
    """
    Forward Kalman filter + backward RTS smoother for a single station.

    State: x = [kt, bias], where bias represents slow pyranometer drift.

    Parameters
    ----------
    observations : np.ndarray (T,)
        Predicted kt values (may contain NaN for missing predictions).
    q_kt : float
        Process noise for kt state (default from HPARAMS).
    q_bias : float
        Process noise for bias state (default from HPARAMS).
    r : float
        Observation noise (default from HPARAMS).

    Returns
    -------
    kt_smooth : np.ndarray (T,) -- RTS-smoothed kt values
    bias_smooth : np.ndarray (T,) -- estimated bias trajectory
    """
    q_kt = q_kt or HPARAMS.get('rts_q_kt', 0.0018)
    q_bias = q_bias or HPARAMS.get('rts_q_bias', 0.00008)
    r = r or HPARAMS.get('rts_r', 0.012)

    n = len(observations)
    if n == 0:
        return np.array([]), np.array([])

    # State-space matrices
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float64)

    H = np.array([[1.0, 0.0]], dtype=np.float64)

    Q = np.array([[q_kt, 0.0],
                  [0.0, q_bias]], dtype=np.float64)

    R = np.array([[r]], dtype=np.float64)

    # Allocate arrays
    x_pred = np.zeros((n, 2), dtype=np.float64)
    P_pred = np.zeros((n, 2, 2), dtype=np.float64)
    x_filt = np.zeros((n, 2), dtype=np.float64)
    P_filt = np.zeros((n, 2, 2), dtype=np.float64)

    # Initialize: first valid observation as kt, bias=0
    first_val = observations[0] if not np.isnan(observations[0]) else 0.5
    x_prev = np.array([first_val, 0.0], dtype=np.float64)
    P_prev = np.eye(2, dtype=np.float64)

    # ---- FORWARD KALMAN FILTER ----
    for t in range(n):
        # Prediction
        x_pred[t] = F @ x_prev
        P_pred[t] = F @ P_prev @ F.T + Q

        # Update (skip if observation is NaN)
        if not np.isnan(observations[t]):
            # Innovation
            y = observations[t] - (H @ x_pred[t])[0]

            # Innovation covariance
            S = (H @ P_pred[t] @ H.T + R)[0, 0]

            # Kalman gain
            K = (P_pred[t] @ H.T) / S  # (2, 1)

            x_filt[t] = x_pred[t] + (K[:, 0] * y)
            P_filt[t] = (np.eye(2) - np.outer(K[:, 0], H[0])) @ P_pred[t]
        else:
            # No observation: prediction = filtered
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]

        x_prev = x_filt[t]
        P_prev = P_filt[t]

    # ---- BACKWARD RTS SMOOTHER ----
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in reversed(range(n - 1)):
        # Smoother gain
        try:
            C = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            # Singular matrix -- skip this step
            continue

        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

    kt_smooth = x_smooth[:, 0].astype(np.float32)
    bias_smooth = x_smooth[:, 1].astype(np.float32)

    return kt_smooth, bias_smooth


def apply_savgol_filter(kt_values, clear_sky_ghi=None,
                        window_length=None, polyorder=None):
    """
    Apply Savitzky-Golay filter to kt predictions.
    Only applies during daylight hours (clear_sky_ghi > 20 W/m2).

    Parameters
    ----------
    kt_values : np.ndarray (T,)
        kt values (typically after RTS smoothing).
    clear_sky_ghi : np.ndarray (T,) or None
        Clear-sky GHI for daylight masking.
    window_length : int
        SavGol window length (default from HPARAMS).
    polyorder : int
        SavGol polynomial order (default from HPARAMS).

    Returns
    -------
    kt_filtered : np.ndarray (T,)
    """
    window_length = window_length or HPARAMS.get('savgol_window', 9)
    polyorder = polyorder or HPARAMS.get('savgol_polyorder', 2)

    kt_filtered = kt_values.copy()

    # Only filter if we have enough data
    if len(kt_values) < window_length:
        return kt_filtered

    if clear_sky_ghi is not None:
        # Apply only during daylight (clear_sky_ghi > 20 W/m2)
        daylight_mask = clear_sky_ghi > 20.0

        if daylight_mask.sum() >= window_length:
            # Extract daylight values, filter, then replace
            daylight_vals = kt_values[daylight_mask]
            filtered = savgol_filter(
                daylight_vals, window_length=window_length,
                polyorder=polyorder, mode='interp'
            )
            kt_filtered[daylight_mask] = filtered.astype(np.float32)
    else:
        # Apply globally
        kt_filtered = savgol_filter(
            kt_values, window_length=window_length,
            polyorder=polyorder, mode='interp'
        ).astype(np.float32)

    return kt_filtered


def postprocess_predictions(df_preds, station_col='station',
                            time_col='timestamp',
                            kt_col='kt_pred',
                            clearsky_col='clear_sky_ghi',
                            is_night_col='is_night',
                            use_rts=True, use_savgol=True):
    """
    Full post-processing pipeline for predictions.

    Pipeline order:
      1. RTS smoother (per station)
      2. Savitzky-Golay filter (daylight only)
      3. Physics gate: GHI = kt * clear_sky, clamp >= 0, night gate

    Parameters
    ----------
    df_preds : pd.DataFrame
        Must contain columns for station, timestamp, kt predictions,
        clear_sky_ghi, and is_night.
    use_rts : bool
        Apply RTS smoother (default True).
    use_savgol : bool
        Apply Savitzky-Golay filter (default True).

    Returns
    -------
    df_preds : pd.DataFrame with added columns:
        'kt_postprocessed', 'ghi_postprocessed', 'bias_estimate'
    """
    import pandas as pd

    df_preds = df_preds.sort_values([station_col, time_col]).copy()

    kt_final = df_preds[kt_col].values.copy().astype(np.float32)
    bias_all = np.zeros(len(df_preds), dtype=np.float32)

    # Process each station independently
    stations = df_preds[station_col].unique()

    for station in stations:
        mask = (df_preds[station_col] == station).values
        kt_station = kt_final[mask].copy()

        if len(kt_station) == 0:
            continue

        # Step 1: RTS Smoother
        if use_rts:
            kt_smooth, bias = rts_smoother_single_station(kt_station)
            kt_station = kt_smooth
            bias_all[mask] = bias

        # Step 2: Savitzky-Golay Filter (AFTER RTS)
        if use_savgol:
            cs_ghi = None
            if clearsky_col in df_preds.columns:
                cs_ghi = df_preds.loc[mask, clearsky_col].values
            kt_station = apply_savgol_filter(kt_station, cs_ghi)

        kt_final[mask] = kt_station

    # Step 3: Physics gates
    kt_final = np.clip(kt_final, 0.0, HPARAMS.get('kt_max', 1.5))

    # Reconstruct GHI
    ghi = kt_final * df_preds[clearsky_col].values.astype(np.float32)
    ghi = np.maximum(ghi, 0.0)

    # Night gate
    if is_night_col in df_preds.columns:
        night_mask = df_preds[is_night_col].values > 0.5
        ghi[night_mask] = 0.0

    df_preds['kt_postprocessed'] = kt_final
    df_preds['ghi_postprocessed'] = ghi
    df_preds['bias_estimate'] = bias_all

    # Summary statistics
    print(f"[POSTPROCESS] Applied to {len(stations)} stations:")
    print(f"  RTS smoother: {'ON' if use_rts else 'OFF'}")
    print(f"  SavGol filter: {'ON' if use_savgol else 'OFF'}")
    print(f"  kt range: [{kt_final.min():.4f}, {kt_final.max():.4f}]")
    print(f"  GHI range: [{ghi.min():.1f}, {ghi.max():.1f}]")
    print(f"  Mean bias estimate: {bias_all.mean():.6f}")

    return df_preds
