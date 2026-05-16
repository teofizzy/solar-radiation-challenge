"""
Annealing Loss with GHI-space MBE anchor for the Zindi metric alignment.

Multi-AI Consensus Fix:
  - Perplexity: "Removing MBE penalty was a mistake. Add back alpha=0.10-0.20"
  - ChatGPT: "Apply MBE in GHI space, not delta_kt space. lambda=0.005-0.02"
  - Gemini: "This is OneCycleLR warmup + FiLM drift. Wait 4-5 epochs to confirm."
  - NotebookLM: "Successful models use standard MSE + residual stacking for bias"

Resolution:
  We keep clear-sky weighted MSE on delta_kt as the primary term (physics-correct),
  but ADD a weak GHI-space MBE anchor to prevent systematic bias accumulation.
  The MBE penalty is computed on GHI (not kt) because that's the evaluation space.
  
  Weight: lambda_mbe = 0.01 (ChatGPT recommendation: small is enough because
  MBE is a global constraint that shouldn't dominate gradients).
"""

import torch
import torch.nn as nn
import numpy as np


class Stage1Loss(nn.Module):
    """Clear-sky weighted MSE on delta_kt with GHI-space MBE anchor.
    
    Two components:
      1. Primary: Clear-sky weighted MSE on delta_kt (bounded space)
      2. Anchor:  |mean(GHI_pred - GHI_true)| (GHI space, prevents bias drift)
    
    The MBE anchor is critical because:
      - Zindi metric = 0.5 * |MBE| + 0.5 * RMSE
      - Without it, the model minimizes squared errors but accumulates systematic bias
      - Small kt bias * clear_sky_ghi (up to 1000 W/m2) = massive GHI bias
    """
    def __init__(self, lambda_mbe=0.01):
        super().__init__()
        self.lambda_mbe = lambda_mbe

    def forward(self, delta_kt_pred, ghi_pred,
                target_delta_kt, target_ghi,
                cos_zenith, clear_sky_ghi):
        """
        Compute clear-sky weighted MSE on delta_kt + GHI-space MBE anchor.
        
        All inputs are (B,) tensors.
        
        Returns
        -------
        loss : scalar tensor
        metrics : dict of scalar metric values for logging
        """
        # 1. Daytime Mask
        day_mask = (cos_zenith > 0) & (~torch.isnan(target_delta_kt))
        
        if day_mask.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'dkt_wmse': 0.0, 'dkt_mbe': 0.0,
                         'ghi_rmse': 0.0, 'mbe': 0.0, 'abs_mbe': 0.0,
                         'zindi': 0.0, 'mbe_loss': 0.0}

        # 2. Delta kt error (bounded space)
        dkt_err = delta_kt_pred[day_mask] - target_delta_kt[day_mask]
        
        # 3. Clear-sky QUADRATIC weighting
        # Why quadratic? GHI = delta_kt * clear_sky, so GHI_error = dkt_error * cs.
        # GHI-space MSE = sum(dkt_error^2 * cs^2) -- the cs^2 is the natural weighting.
        # Linear weighting (cs/mean) under-weights noon by ~200x vs evaluation.
        # Quadratic weighting (cs^2/mean_cs^2) exactly matches GHI-space MSE geometry.
        cs_ghi_day = clear_sky_ghi[day_mask]
        cs_sq = cs_ghi_day ** 2
        weights = cs_sq / (cs_sq.mean() + 1e-6)
        # No clamp needed: quadratic naturally self-normalizes via mean division
        
        # 4. Primary: Weighted MSE on delta_kt (equivalent to GHI-space MSE)
        wmse = (weights * dkt_err ** 2).mean()
        
        # 5. Anchor: GHI-space MBE penalty (prevents bias drift)
        # Computed on GHI (evaluation space), NOT on delta_kt (training space)
        ghi_err = ghi_pred[day_mask] - target_ghi[day_mask]
        valid_ghi = ~torch.isnan(ghi_err)
        
        if valid_ghi.sum() > 0:
            ghi_err_valid = ghi_err[valid_ghi]
            mbe_ghi = ghi_err_valid.mean()
            mbe_loss = torch.abs(mbe_ghi)
        else:
            mbe_loss = torch.tensor(0.0, device=ghi_pred.device)
            mbe_ghi = torch.tensor(0.0)
        
        # 6. Combined loss
        loss = wmse + self.lambda_mbe * mbe_loss
        
        # 7. Metrics for logging (NOT used in gradient)
        with torch.no_grad():
            if valid_ghi.sum() > 0:
                ghi_rmse = torch.sqrt(torch.mean(ghi_err_valid ** 2) + 1e-8).item()
                ghi_mbe = mbe_ghi.item()
                ghi_abs_mbe = abs(ghi_mbe)
                zindi_score = 0.5 * ghi_abs_mbe + 0.5 * ghi_rmse
            else:
                ghi_rmse = ghi_mbe = ghi_abs_mbe = zindi_score = 0.0
        
        metrics = {
            'loss': loss.item(),
            'dkt_wmse': wmse.item(),
            'dkt_mbe': dkt_err.mean().item(),
            'ghi_rmse': ghi_rmse,
            'mbe': ghi_mbe,
            'abs_mbe': ghi_abs_mbe,
            'zindi': zindi_score,
            'mbe_loss': (self.lambda_mbe * mbe_loss).item(),
        }
        
        return loss, metrics


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
