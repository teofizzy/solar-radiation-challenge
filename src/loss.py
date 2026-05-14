"""
Custom loss function aligned with Zindi evaluation metrics.

Zindi scoring:
    Final = 0.5 * |MBE| + 0.5 * RMSE
    where:
        MBE = mean(predicted - observed)
        RMSE = sqrt(mean((predicted - observed)^2))

Architecture:
    Multi-task loss with two components:
    1. Auxiliary: LogCosh(delta_kt) for stable gradient flow during early training.
    2. Primary: Direct Zindi composite (0.5*RMSE + 0.5*|MBE|) on reconstructed GHI.

    Total = dkt_weight * LogCosh(delta_kt) + zindi_weight * ZindiComposite(GHI)

    Default weights: dkt_weight=0.4, zindi_weight=0.6 (per multi-AI consensus).
"""

import torch
import torch.nn as nn
import numpy as np


class ZindiSolarLoss(nn.Module):
    """
    Multi-task loss directly targeting the Zindi leaderboard metric.
    
    Components:
        1. LogCosh(delta_kt) -- auxiliary, stabilizes early learning.
        2. ZindiComposite(GHI) = 0.5 * RMSE(GHI) + 0.5 * |MBE(GHI)| -- primary.
    
    Parameters
    ----------
    dkt_weight : float
        Weight for the auxiliary delta_kt LogCosh loss.
    zindi_weight : float
        Weight for the direct Zindi composite on reconstructed GHI.
    """
    def __init__(self, dkt_weight: float = 0.4, zindi_weight: float = 0.6):
        super().__init__()
        self.dkt_weight = dkt_weight
        self.zindi_weight = zindi_weight

    def forward(self, delta_kt_pred, ghi_pred,
                target_delta_kt, target_ghi,
                is_night, clear_sky_ghi):
        """
        Compute multi-task loss.
        
        All inputs are (B,) tensors. Daytime-only masking is applied internally.
        """
        # 1. Daytime Mask (exclude night and NaN targets)
        day_mask = (is_night < 0.5) & (~torch.isnan(target_ghi))
        if day_mask.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'dkt_logcosh': 0.0, 'ghi_rmse': 0.0, 
                         'mbe': 0.0, 'zindi': 0.0}

        # 2. Auxiliary: Delta kt LogCosh (numerically stable version)
        # log(cosh(x)) = |x| + log(1 + exp(-2*|x|)) - log(2)
        # This prevents INF overflow when dkt_err is large (e.g. > 88 in float32 or > 11 in float16)
        dkt_err = delta_kt_pred[day_mask] - target_delta_kt[day_mask]
        abs_err = torch.abs(dkt_err)
        loss_dkt = torch.mean(abs_err + torch.nn.functional.softplus(-2.0 * abs_err) - np.log(2.0))
        
        # 3. Primary: Zindi Composite on reconstructed GHI
        ghi_err = ghi_pred[day_mask] - target_ghi[day_mask]
        
        # RMSE (in W/m2, same units as leaderboard)
        rmse = torch.sqrt(torch.mean(ghi_err ** 2) + 1e-8)
        
        # |MBE| (absolute mean bias)
        mbe = torch.mean(ghi_err)
        abs_mbe = torch.abs(mbe)
        
        # Zindi composite: 0.5 * RMSE + 0.5 * |MBE|
        zindi_composite = 0.5 * rmse + 0.5 * abs_mbe
        
        # 4. Scale GHI loss to roughly match delta_kt scale for balanced gradients
        # RMSE is ~50-200 W/m2, LogCosh(dkt) is ~0.01-0.5
        # Scale factor of 1/100 brings them to comparable magnitude
        scaled_zindi = zindi_composite / 100.0
        
        # 5. Total weighted loss
        total_loss = self.dkt_weight * loss_dkt + self.zindi_weight * scaled_zindi
        
        metrics = {
            'loss': total_loss.item(),
            'dkt_logcosh': loss_dkt.item(),
            'ghi_rmse': rmse.item(),
            'mbe': mbe.item(),
            'abs_mbe': abs_mbe.item(),
            'zindi': (0.5 * abs_mbe.item() + 0.5 * rmse.item()),
            'zindi_composite_raw': zindi_composite.item(),
        }
        
        return total_loss, metrics


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
