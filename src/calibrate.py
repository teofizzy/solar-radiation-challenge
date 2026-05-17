"""
Per-station multiplicative ratio calibration.

Forces MBE -> 0 per station by computing ratio = sum(true) / sum(pred)
on validation data, then applying to test predictions.

Multi-AI consensus:
    - Apply to FINAL stacked output only (not before stacking)
    - Use sum ratio (not mean ratio) for stability with GHI=0 nights
    - Regularize per-station weights toward global (L2 penalty)

References:
    - Colleague pipeline: "mathematically guaranteed MBE = 0"
    - Perplexity: "Zindi metric 0.5*|MBE| -> MBE=0 saves 50% of score"
    - ChatGPT: "calibrate after stacking preserves ensemble diversity"
"""

import numpy as np
from scipy.optimize import minimize_scalar


def compute_station_ratios(y_true, y_pred, station_ids, min_samples=50):
    """
    Compute per-station multiplicative calibration ratios.

    Uses validation data ONLY (no leakage).
    ratio = sum(true[station]) / sum(pred[station])

    Parameters
    ----------
    y_true : np.ndarray (N,)
        True GHI values from validation set.
    y_pred : np.ndarray (N,)
        Stacked model predictions on validation set.
    station_ids : np.ndarray (N,)
        Station identifiers.
    min_samples : int
        Minimum daytime samples per station (default: 50).

    Returns
    -------
    ratios : dict
        {station_id: calibration_ratio}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ratios = {}

    for station in np.unique(station_ids):
        mask = (station_ids == station) & (~np.isnan(y_true)) & (y_pred > 1.0)

        if mask.sum() >= min_samples:
            ratios[station] = y_true[mask].sum() / y_pred[mask].sum()
        else:
            ratios[station] = 1.0

    # Log summary
    ratio_values = list(ratios.values())
    print(f"[CALIBRATE] Computed ratios for {len(ratios)} stations:")
    print(f"  Mean ratio: {np.mean(ratio_values):.4f}")
    print(f"  Std ratio:  {np.std(ratio_values):.4f}")
    print(f"  Range:      [{np.min(ratio_values):.4f}, {np.max(ratio_values):.4f}]")

    return ratios


def apply_calibration(y_pred, station_ids, ratios):
    """
    Apply per-station multiplicative calibration to predictions.

    Parameters
    ----------
    y_pred : np.ndarray (N,)
        Model predictions to calibrate.
    station_ids : np.ndarray (N,)
        Station identifiers.
    ratios : dict
        {station_id: calibration_ratio} from compute_station_ratios.

    Returns
    -------
    y_cal : np.ndarray (N,)
        Calibrated predictions (GHI >= 0).
    """
    y_cal = np.asarray(y_pred, dtype=np.float64).copy()

    for station in np.unique(station_ids):
        mask = station_ids == station
        y_cal[mask] = y_pred[mask] * ratios.get(station, 1.0)

    y_cal = np.maximum(y_cal, 0.0)

    return y_cal.astype(np.float32)


def optimize_station_weights(bilstm_preds, lgbm_preds, y_true,
                             station_ids, global_w_bi=0.4, lambda_reg=10.0):
    """
    Optimize per-station stacking weights with L2 regularization toward global.

    For each station, finds optimal w_bi that minimizes:
        Zindi(w_bi * BiLSTM + (1-w_bi) * LightGBM, y_true) + lambda * (w_bi - global_w)^2

    Parameters
    ----------
    bilstm_preds : np.ndarray (N,)
        BiLSTM GHI predictions on validation set.
    lgbm_preds : np.ndarray (N,)
        LightGBM GHI predictions on validation set.
    y_true : np.ndarray (N,)
        True GHI values.
    station_ids : np.ndarray (N,)
        Station identifiers.
    global_w_bi : float
        Global BiLSTM weight prior (default: 0.4).
    lambda_reg : float
        L2 regularization strength toward global (default: 10.0).

    Returns
    -------
    station_weights : dict
        {station_id: optimal_w_bi}
    global_w : float
        Global optimal weight (for fallback).
    """
    bilstm_preds = np.asarray(bilstm_preds, dtype=np.float64)
    lgbm_preds = np.asarray(lgbm_preds, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    station_weights = {}

    for station in np.unique(station_ids):
        mask = (station_ids == station) & (~np.isnan(y_true))

        if mask.sum() < 100:
            # Not enough samples -- use global weight
            station_weights[station] = global_w_bi
            continue

        y_bi = bilstm_preds[mask]
        y_lgb = lgbm_preds[mask]
        y_t = y_true[mask]

        def objective(w_bi):
            w_lgb = 1.0 - w_bi
            pred = w_bi * y_bi + w_lgb * y_lgb
            residuals = pred - y_t
            mbe = np.abs(np.mean(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
            zindi = 0.5 * mbe + 0.5 * rmse
            penalty = lambda_reg * (w_bi - global_w_bi) ** 2
            return zindi + penalty

        result = minimize_scalar(objective, bounds=(0.1, 0.9), method='bounded')
        station_weights[station] = float(result.x)

    # Log
    weights = list(station_weights.values())
    print(f"[WEIGHTS] Per-station BiLSTM weights:")
    print(f"  Mean: {np.mean(weights):.3f}")
    print(f"  Std:  {np.std(weights):.3f}")
    print(f"  Range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")

    return station_weights, global_w_bi


def apply_station_weights(bilstm_preds, lgbm_preds, station_ids,
                          station_weights, default_w_bi=0.4):
    """
    Apply per-station stacking weights to predictions.

    Parameters
    ----------
    bilstm_preds : np.ndarray (N,)
        BiLSTM predictions.
    lgbm_preds : np.ndarray (N,)
        LightGBM predictions.
    station_ids : np.ndarray (N,)
        Station identifiers.
    station_weights : dict
        {station_id: w_bi} from optimize_station_weights.
    default_w_bi : float
        Fallback weight for unknown stations.

    Returns
    -------
    stacked : np.ndarray (N,)
    """
    stacked = np.zeros(len(bilstm_preds), dtype=np.float64)

    for station in np.unique(station_ids):
        mask = station_ids == station
        w_bi = station_weights.get(station, default_w_bi)
        w_lgb = 1.0 - w_bi
        stacked[mask] = w_bi * bilstm_preds[mask] + w_lgb * lgbm_preds[mask]

    return np.maximum(stacked, 0.0).astype(np.float32)
