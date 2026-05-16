"""
Stage 1 Loss: Clear-Sky Weighted MSE on delta_kt.

Design rationale (from multi-AI consensus + Gemini correction):
    - MSE on delta_kt keeps optimization in bounded kt-space, preventing 
      the gradient explosion that occurs with unbounded GHI-space targets.
    - Clear-sky weighting addresses Gemini's valid critique: a 0.1 kt error 
      at dawn (GHI_cs=50, 5 W/m2) should cost less than at noon (GHI_cs=1000, 100 W/m2).
    - Daytime-only masking eliminates nighttime noise from the loss landscape.
    - No multi-task components. No signed MBE penalty. Single objective.

Zindi evaluation function preserved for validation/submission scoring.
"""

import torch
import torch.nn as nn
import numpy as np


class Stage1Loss(nn.Module):
    """Clear-sky weighted MSE on delta_kt.
    
    The weight is proportional to clear_sky_ghi, so errors at solar noon
    (high GHI_cs) are penalized proportionally more than errors at dawn/dusk.
    This aligns the kt-space optimization with the Zindi GHI-space metric.
    
    Parameters
    ----------
    None. This is a single-objective loss with no tunable weights.
    """
    def __init__(self):
        super().__init__()

    def forward(self, delta_kt_pred, ghi_pred,
                target_delta_kt, target_ghi,
                cos_zenith, clear_sky_ghi):
        """
        Compute clear-sky weighted MSE on delta_kt.
        
        All inputs are (B,) tensors. Daytime masking uses cos_zenith > 0 
        (replaces is_night binary flag).
        
        Parameters
        ----------
        delta_kt_pred : (B,) predicted clearness index correction
        ghi_pred : (B,) reconstructed GHI (for metric logging only)
        target_delta_kt : (B,) true clearness index correction
        target_ghi : (B,) true GHI (for metric logging only)
        cos_zenith : (B,) cosine of solar zenith angle
        clear_sky_ghi : (B,) clear-sky GHI at center timestep
        
        Returns
        -------
        loss : scalar tensor
        metrics : dict of scalar metric values for logging
        """
        # 1. Daytime Mask: cos_zenith > 0 means sun above horizon (SZA < 90)
        # Also exclude NaN targets (test samples that leaked into training)
        day_mask = (cos_zenith > 0) & (~torch.isnan(target_delta_kt))
        
        if day_mask.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'dkt_wmse': 0.0, 'dkt_mbe': 0.0,
                         'ghi_rmse': 0.0, 'mbe': 0.0, 'zindi': 0.0}

        # 2. Delta kt error (bounded space)
        dkt_err = delta_kt_pred[day_mask] - target_delta_kt[day_mask]
        
        # 3. Clear-sky weighting (Gemini correction)
        # Normalize so that mean weight = 1.0 (loss magnitude is stable across batches)
        cs_ghi_day = clear_sky_ghi[day_mask]
        weights = cs_ghi_day / (cs_ghi_day.mean() + 1e-6)
        # Clamp weights to prevent extreme outliers from dominating
        weights = torch.clamp(weights, min=0.1, max=5.0)
        
        # 4. Weighted MSE on delta_kt
        wmse = (weights * dkt_err ** 2).mean()
        
        # 5. Compute GHI-space metrics for logging (NOT used in loss gradient)
        with torch.no_grad():
            ghi_err = ghi_pred[day_mask] - target_ghi[day_mask]
            valid_ghi = ~torch.isnan(ghi_err)
            if valid_ghi.sum() > 0:
                ghi_err_valid = ghi_err[valid_ghi]
                ghi_rmse = torch.sqrt(torch.mean(ghi_err_valid ** 2) + 1e-8).item()
                ghi_mbe = torch.mean(ghi_err_valid).item()
                ghi_abs_mbe = abs(ghi_mbe)
                zindi_score = 0.5 * ghi_abs_mbe + 0.5 * ghi_rmse
            else:
                ghi_rmse = ghi_mbe = ghi_abs_mbe = zindi_score = 0.0
        
        metrics = {
            'loss': wmse.item(),
            'dkt_wmse': wmse.item(),
            'dkt_mbe': dkt_err.mean().item(),
            'ghi_rmse': ghi_rmse,
            'mbe': ghi_mbe,
            'abs_mbe': ghi_abs_mbe,
            'zindi': zindi_score,
        }
        
        return wmse, metrics


def compute_zindi_score(ghi_pred, ghi_target):
    """
    Compute the exact Zindi leaderboard score (numpy).

    Parameters
    ----------
    ghi_pred : array-like
        Predicted radiation values.
    ghi_target : array-like
        True radiation values.

    Returns
    -------
    float : 0.5 * |MBE| + 0.5 * RMSE
    """
    pred = np.array(ghi_pred, dtype=np.float64)
    target = np.array(ghi_target, dtype=np.float64)

    valid = ~np.isnan(target) & ~np.isnan(pred)
    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return float('inf')

    residuals = pred - target
    mbe = np.abs(np.mean(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))

    return 0.5 * mbe + 0.5 * rmse
