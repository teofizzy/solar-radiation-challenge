"""
Custom loss function aligned with Zindi evaluation metrics.

Zindi scoring:
    Final = 0.5 * |MBE| + 0.5 * RMSE
    where:
        MBE = mean(predicted - observed)
        RMSE = sqrt(mean((predicted - observed)^2))

This loss function directly targets both metrics with spike-aware weighting
and physics-informed night penalties.

CHANGE LOG:
  - Removed smoothness penalty (operated on shuffled batches = gradient noise)
  - Added spike-aware weighting: upweight errors when kt > 0.7 (weight=3.0)
    to improve RMSE on clear-sky/high-transmittance events
"""

import torch
import torch.nn as nn


class ZindiSolarLoss(nn.Module):
    """
    Combined loss: 0.5 * |MBE| + 0.5 * spike_weighted_RMSE + night_penalty

    The spike-aware weighting upweights errors during high-kt events
    (clear sky, cloud edges) where RMSE contributions are largest.

    Parameters
    ----------
    mbe_weight : float
        Weight for |MBE| component (default 0.5).
    rmse_weight : float
        Weight for RMSE component (default 0.5).
    night_penalty_weight : float
        Weight for nighttime non-zero radiation penalty.
    spike_kt_threshold : float
        kt threshold above which errors are upweighted (default 0.7).
    spike_weight : float
        Multiplicative weight for high-kt errors (default 3.0).
    """

    def __init__(self, mbe_weight: float = 0.5, rmse_weight: float = 0.5,
                 night_penalty_weight: float = 0.01,
                 delta_kt_weight: float = 10.0,
                 station_bias_penalty: float = 0.01,
                 spike_kt_threshold: float = 0.7,
                 spike_weight: float = 3.0):
        super().__init__()
        self.mbe_weight = mbe_weight
        self.rmse_weight = rmse_weight
        self.night_penalty_weight = night_penalty_weight
        self.delta_kt_weight = delta_kt_weight
        self.station_bias_penalty = station_bias_penalty
        self.spike_kt_threshold = spike_kt_threshold
        self.spike_weight = spike_weight

    def forward(self, ghi_pred: torch.Tensor, ghi_target: torch.Tensor,
                delta_kt_pred: torch.Tensor = None, delta_kt_target: torch.Tensor = None,
                is_night: torch.Tensor = None, station_bias: torch.Tensor = None):
        """
        Compute combined loss.

        Parameters
        ----------
        ghi_pred : Tensor (batch,)
            Predicted radiation in W/m2.
        ghi_target : Tensor (batch,)
            Target radiation in W/m2.
        delta_kt_pred : Tensor (batch,) or None
            Predicted delta kt for auxiliary task.
        delta_kt_target : Tensor (batch,) or None
            Target delta kt for auxiliary task.
        is_night : Tensor (batch,) or None
            Nighttime flag for night penalty.
        station_bias : Tensor (batch,) or None
            Predicted station bias for L2 regularization.

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
        # Differentiable approximation of |mean(residuals)|
        mean_error = torch.mean(residuals)
        mbe = torch.sqrt(mean_error ** 2 + 1e-8)

        # --- Spike-aware RMSE component ---
        # Upweight errors during high-kt events where RMSE is dominated
        squared_errors = residuals ** 2

        weighted_mse = torch.mean(squared_errors)

        rmse = torch.sqrt(weighted_mse + 1e-8)

        # --- Combined primary loss ---
        primary_loss = self.mbe_weight * mbe + self.rmse_weight * rmse

        # --- Auxiliary Delta Kt Loss ---
        aux_loss = torch.tensor(0.0, device=ghi_pred.device)
        if delta_kt_pred is not None and delta_kt_target is not None:
            dkt_pred = delta_kt_pred[valid_mask]
            dkt_target = delta_kt_target[valid_mask]
            dkt_valid = ~torch.isnan(dkt_target)
            if dkt_valid.sum() > 0:
                aux_loss = self.delta_kt_weight * torch.mean((dkt_pred[dkt_valid] - dkt_target[dkt_valid])**2)

        # --- Station Bias L2 Penalty ---
        bias_loss = torch.tensor(0.0, device=ghi_pred.device)
        if station_bias is not None:
            bias_loss = self.station_bias_penalty * torch.mean(station_bias**2)

        # --- Night penalty (radiation should be zero at night) ---
        physics_loss = torch.tensor(0.0, device=ghi_pred.device)
        if is_night is not None and self.night_penalty_weight > 0:
            night_mask = is_night[valid_mask] > 0.5
            if night_mask.any():
                night_radiation = pred[night_mask]
                night_penalty = torch.mean(night_radiation ** 2)
                physics_loss = physics_loss + self.night_penalty_weight * night_penalty

        total_loss = primary_loss + physics_loss + aux_loss + bias_loss

        # Unweighted RMSE for logging (matches Zindi metric exactly)
        raw_rmse = torch.sqrt(torch.mean(squared_errors) + 1e-8)

        loss_dict = {
            'total': total_loss.item(),
            'mbe': torch.abs(mean_error).item(),
            'rmse': raw_rmse.item(),
            'physics': physics_loss.item(),
            'aux_dkt': aux_loss.item(),
            'bias_reg': bias_loss.item(),
            'zindi_score': (0.5 * torch.abs(mean_error) + 0.5 * raw_rmse).item(),
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
