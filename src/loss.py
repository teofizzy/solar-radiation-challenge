"""
Direct Zindi Loss for solar radiation reconstruction.

Computes the exact Zindi leaderboard metric as the training loss:
    loss = 0.5 * |MBE| + 0.5 * RMSE

Where:
    MBE  = mean(ghi_pred - ghi_true)  (Mean Bias Error in W/m2)
    RMSE = sqrt(mean((ghi_pred - ghi_true)^2))  (Root Mean Squared Error in W/m2)

Proven in solar-sweep-1 (Zindi=45.48, MBE=2.12):
    - Direct optimization of leaderboard metric was the KEY differentiator
    - Huber loss caused MBE blowup (2.12 -> 9.06) because it ignores bias direction
    - Perplexity: "Direct optimization of leaderboard metric was HUGE"
    - ChatGPT: "No proxy mismatch, no latent reconstruction instability"
"""

import torch
import torch.nn as nn
import numpy as np


class ZindiLoss(nn.Module):
    """Direct Zindi metric loss: 0.5 * |MBE| + 0.5 * RMSE on GHI.

    Operates entirely in GHI space (W/m2), which is the evaluation space.
    No proxy losses, no delta_kt indirection, no clear-sky weighting.

    Additional physics regularization:
        - kt smoothness penalty (optional, weight=0.001)
        - Night penalty (should be zero, catches leakage)
    """
    def __init__(self, lambda_smooth=0.001, lambda_night=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_night = lambda_night

    def forward(self, kt_pred, ghi_pred, target_ghi, is_night):
        """
        Compute direct Zindi loss on GHI predictions.

        Parameters
        ----------
        kt_pred : (B,) tensor -- predicted clearness index
        ghi_pred : (B,) tensor -- predicted GHI (W/m2)
        target_ghi : (B,) tensor -- observed GHI (W/m2), NaN for test
        is_night : (B,) tensor -- binary nighttime flag

        Returns
        -------
        loss : scalar tensor (differentiable)
        metrics : dict of scalar metric values for logging
        """
        # 1. Filter to valid daytime samples (non-NaN targets, daytime)
        valid = (~torch.isnan(target_ghi)) & (is_night < 0.5)

        if valid.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'ghi_rmse': 0.0, 'mbe': 0.0,
                         'abs_mbe': 0.0, 'zindi': 0.0}

        ghi_p = ghi_pred[valid]
        ghi_t = target_ghi[valid]
        residuals = ghi_p - ghi_t

        # 2. MBE component: |mean(residuals)|
        mbe = residuals.mean()
        abs_mbe = torch.abs(mbe)

        # 3. RMSE component: sqrt(mean(residuals^2))
        # Add small epsilon inside sqrt for gradient stability at zero
        rmse = torch.sqrt(torch.mean(residuals ** 2) + 1e-8)

        # 4. Zindi score = 0.5 * |MBE| + 0.5 * RMSE
        zindi_loss = 0.5 * abs_mbe + 0.5 * rmse

        # 5. Physics regularization (optional, very weak)
        loss = zindi_loss

        # kt smoothness: penalize extreme kt values (should be smooth)
        if self.lambda_smooth > 0:
            kt_valid = kt_pred[valid]
            # Penalize kt very far from 0.5 (helps convergence, not physics)
            kt_penalty = torch.mean((kt_valid - 0.5) ** 2)
            loss = loss + self.lambda_smooth * kt_penalty

        # Night penalty: catch any night leakage past the hard gate
        if self.lambda_night > 0:
            night_mask = is_night > 0.5
            if night_mask.any():
                night_ghi = ghi_pred[night_mask]
                night_penalty = torch.mean(night_ghi ** 2)
                loss = loss + self.lambda_night * night_penalty

        # 6. Metrics for logging (no gradient)
        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'ghi_rmse': rmse.item(),
                'mbe': mbe.item(),
                'abs_mbe': abs_mbe.item(),
                'zindi': zindi_loss.item(),
            }

        return loss, metrics


class SolarHuberLoss(nn.Module):
    """Huber loss on GHI (daytime only) with physics regularization.

    KEPT for potential LightGBM Stage 2 training or future experimentation.
    NOT used for BiLSTM training (Zindi loss is superior for this metric).
    """
    def __init__(self, delta=50.0, lambda_night=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.lambda_night = lambda_night
        self.delta = delta

    def forward(self, kt_pred, ghi_pred, target_ghi, is_night):
        valid = (~torch.isnan(target_ghi)) & (is_night < 0.5)

        if valid.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'ghi_rmse': 0.0, 'mbe': 0.0,
                         'abs_mbe': 0.0, 'zindi': 0.0}

        ghi_p = ghi_pred[valid]
        ghi_t = target_ghi[valid]

        loss = self.huber(ghi_p, ghi_t)

        if self.lambda_night > 0:
            night_mask = is_night > 0.5
            if night_mask.any():
                night_ghi = ghi_pred[night_mask]
                night_penalty = torch.mean(night_ghi ** 2)
                loss = loss + self.lambda_night * night_penalty

        with torch.no_grad():
            residuals = ghi_p - ghi_t
            mbe = torch.mean(residuals)
            abs_mbe = torch.abs(mbe)
            rmse = torch.sqrt(torch.mean(residuals ** 2) + 1e-8)
            zindi = 0.5 * abs_mbe + 0.5 * rmse

            metrics = {
                'loss': loss.item(),
                'ghi_rmse': rmse.item(),
                'mbe': mbe.item(),
                'abs_mbe': abs_mbe.item(),
                'zindi': zindi.item(),
            }

        return loss, metrics


def compute_zindi_score(ghi_pred, ghi_target):
    """
    Compute the exact Zindi leaderboard score (numpy).

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
