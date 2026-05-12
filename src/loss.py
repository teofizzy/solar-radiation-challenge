"""
Custom loss function aligned with Zindi evaluation metrics.

Zindi scoring:
    Final = 0.5 * |MBE| + 0.5 * RMSE
    where:
        MBE = mean(predicted - observed)
        RMSE = sqrt(mean((predicted - observed)^2))

This loss function directly targets both metrics with optional physics penalties.
"""

import torch
import torch.nn as nn


class ZindiSolarLoss(nn.Module):
    """
    Combined loss: 0.5 * |MBE| + 0.5 * RMSE + physics_penalties

    Parameters
    ----------
    mbe_weight : float
        Weight for |MBE| component (default 0.5).
    rmse_weight : float
        Weight for RMSE component (default 0.5).
    smoothness_weight : float
        Weight for kt temporal smoothness penalty.
    night_penalty_weight : float
        Weight for nighttime non-zero radiation penalty.
    """

    def __init__(self, mbe_weight: float = 0.5, rmse_weight: float = 0.5,
                 smoothness_weight: float = 0.02,
                 night_penalty_weight: float = 0.01):
        super().__init__()
        self.mbe_weight = mbe_weight
        self.rmse_weight = rmse_weight
        self.smoothness_weight = smoothness_weight
        self.night_penalty_weight = night_penalty_weight

    def forward(self, ghi_pred: torch.Tensor, ghi_target: torch.Tensor,
                kt_pred: torch.Tensor = None, is_night: torch.Tensor = None):
        """
        Compute combined loss.

        Parameters
        ----------
        ghi_pred : Tensor (batch,)
            Predicted radiation in W/m2.
        ghi_target : Tensor (batch,)
            Target radiation in W/m2.
        kt_pred : Tensor (batch,) or None
            Predicted clearness index for smoothness penalty.
        is_night : Tensor (batch,) or None
            Nighttime flag for night penalty.

        Returns
        -------
        total_loss : Tensor (scalar)
        loss_dict : dict of individual loss components
        """
        # Filter valid (non-NaN) targets
        valid_mask = ~torch.isnan(ghi_target)
        if valid_mask.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'total': 0.0, 'mbe': 0.0, 'rmse': 0.0}

        pred = ghi_pred[valid_mask]
        target = ghi_target[valid_mask]

        residuals = pred - target

        # --- MBE component ---
        # |mean(residuals)| -- penalizes systematic bias
        mbe = torch.abs(torch.mean(residuals))

        # --- RMSE component ---
        # sqrt(mean(residuals^2)) -- penalizes large errors
        rmse = torch.sqrt(torch.mean(residuals ** 2) + 1e-8)

        # --- Combined primary loss ---
        primary_loss = self.mbe_weight * mbe + self.rmse_weight * rmse

        # --- Physics penalties ---
        physics_loss = torch.tensor(0.0, device=ghi_pred.device)

        # Smoothness penalty on kt (cloud transmittance is physically continuous)
        if kt_pred is not None and self.smoothness_weight > 0:
            kt_valid = kt_pred[valid_mask]
            if len(kt_valid) > 1:
                kt_diff = torch.diff(kt_valid)
                smoothness = torch.mean(kt_diff ** 2)
                physics_loss = physics_loss + self.smoothness_weight * smoothness

        # Night penalty (radiation should be zero at night)
        if is_night is not None and self.night_penalty_weight > 0:
            night_mask = is_night[valid_mask] > 0.5
            if night_mask.any():
                night_radiation = pred[night_mask]
                night_penalty = torch.mean(night_radiation ** 2)
                physics_loss = physics_loss + self.night_penalty_weight * night_penalty

        total_loss = primary_loss + physics_loss

        loss_dict = {
            'total': total_loss.item(),
            'mbe': mbe.item(),
            'rmse': rmse.item(),
            'physics': physics_loss.item(),
            'zindi_score': (0.5 * mbe + 0.5 * rmse).item(),
        }

        return total_loss, loss_dict


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
    import numpy as np
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
