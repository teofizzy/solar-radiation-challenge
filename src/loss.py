"""
Loss functions for the Hybrid BiLSTM + LightGBM pipeline.

Training loss: SolarHuberLoss (Huber on GHI, daytime only)
Evaluation metric: ZindiLoss (0.5 * |MBE| + 0.5 * RMSE) -- NOT for training

Multi-AI Consensus:
    - NotebookLM [1-3]: "MSE/Huber for training, Zindi for evaluation"
    - ChatGPT: "Huber or MSE on residuals, NOT direct Zindi loss"
    - Gemini: "Huber loss -- robust to outliers in meteorological data"
    - Perplexity: "Direct Zindi loss causes MBE/RMSE oscillation"

Design:
    - Huber delta is sweepable (default=50 W/m2, range [30, 100])
    - Large delta -> behaves like MSE (penalizes large errors more)
    - Small delta -> behaves like MAE (robust to outliers)
    - MBE correction done via per-station ratio calibration, NOT loss function
"""

import torch
import torch.nn as nn
import numpy as np


class SolarHuberLoss(nn.Module):
    """Huber loss on GHI (daytime only) with physics regularization.

    Operates in GHI space (W/m2). Huber provides robustness to
    sensor noise / dust storm outliers while maintaining smooth gradients.

    Parameters
    ----------
    delta : float
        Huber transition point (W/m2). Sweepable via HPARAMS.
    lambda_night : float
        Penalty weight for nighttime GHI leakage.
    """
    def __init__(self, delta=50.0, lambda_night=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.lambda_night = lambda_night
        self.delta = delta

    def forward(self, residual_pred, ghi_pred, target_ghi, is_night):
        """
        Compute Huber loss on GHI predictions (daytime only).

        Parameters
        ----------
        residual_pred : (B,) tensor -- predicted residual (GHI - MDSSF)
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

        # 2. Huber loss on GHI (main training loss)
        loss = self.huber(ghi_p, ghi_t)

        # 3. Night penalty: catch any night leakage past the hard gate
        if self.lambda_night > 0:
            night_mask = is_night > 0.5
            if night_mask.any():
                night_ghi = ghi_pred[night_mask]
                night_penalty = torch.mean(night_ghi ** 2)
                loss = loss + self.lambda_night * night_penalty

        # 4. Metrics for logging (no gradient)
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


class ZindiLoss(nn.Module):
    """Zindi metric: 0.5 * |MBE| + 0.5 * RMSE on GHI.

    EVALUATION ONLY -- not used for training (causes oscillation).
    Kept for validation scoring and sweep objective.
    """
    def forward(self, residual_pred, ghi_pred, target_ghi, is_night):
        valid = (~torch.isnan(target_ghi)) & (is_night < 0.5)

        if valid.sum() == 0:
            zero = torch.tensor(0.0, device=ghi_pred.device, requires_grad=True)
            return zero, {'loss': 0.0, 'ghi_rmse': 0.0, 'mbe': 0.0,
                         'abs_mbe': 0.0, 'zindi': 0.0}

        ghi_p = ghi_pred[valid]
        ghi_t = target_ghi[valid]
        residuals = ghi_p - ghi_t

        mbe = residuals.mean()
        abs_mbe = torch.abs(mbe)
        rmse = torch.sqrt(torch.mean(residuals ** 2) + 1e-8)
        zindi_loss = 0.5 * abs_mbe + 0.5 * rmse

        with torch.no_grad():
            metrics = {
                'loss': zindi_loss.item(),
                'ghi_rmse': rmse.item(),
                'mbe': mbe.item(),
                'abs_mbe': abs_mbe.item(),
                'zindi': zindi_loss.item(),
            }

        return zindi_loss, metrics


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
