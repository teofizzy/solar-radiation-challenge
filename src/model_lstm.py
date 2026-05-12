"""
Physics-Informed BiLSTM for solar radiation reconstruction.
Predicts clearness index kt in (0, 1), then reconstructs GHI = kt * clear_sky_ghi.

Architecture:
    Input (batch, seq_len, n_features + embed_dim)
      -> LayerNorm
      -> BiLSTM (2 layers, 128 hidden, bidirectional)
      -> Center hidden state extraction (256-dim)
      -> Linear(256, 64) -> GELU -> Dropout
      -> Linear(64, 1) -> Sigmoid (kt in 0..1)
      -> GHI = kt * clear_sky_ghi
      -> Night gate: GHI * (1 - is_night)
"""

import torch
import torch.nn as nn

from src.config import HPARAMS


class PhysicsInformedBiLSTM(nn.Module):
    """
    Bi-directional LSTM that predicts clearness index (kt) for solar radiation
    reconstruction. Physics constraints are enforced via:
      1. Sigmoid output: kt bounded in (0, 1)
      2. Night gate: GHI forced to 0 when solar_zenith > 90
      3. Station embedding: global model conditioned on station identity

    Parameters
    ----------
    n_features : int
        Number of input covariate features per timestep.
    n_stations : int
        Number of unique stations for embedding.
    hidden_dim : int
        LSTM hidden dimension (per direction).
    n_layers : int
        Number of LSTM layers.
    embed_dim : int
        Station embedding dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_features: int, n_stations: int,
                 hidden_dim: int = None, n_layers: int = None,
                 embed_dim: int = None, dropout: float = None):
        super().__init__()

        hidden_dim = hidden_dim or HPARAMS['hidden_dim']
        n_layers = n_layers or HPARAMS['n_layers']
        embed_dim = embed_dim or HPARAMS['embed_dim']
        dropout = dropout or HPARAMS['dropout']

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.half_window = HPARAMS['half_window']

        # Station embedding
        self.station_embedding = nn.Embedding(n_stations, embed_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features + embed_dim)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=n_features + embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Output head: predicts kt from center hidden state
        # BiLSTM output dim = 2 * hidden_dim
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # kt in (0, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers, orthogonal for LSTM."""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for module in self.output_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor,
                clear_sky_ghi: torch.Tensor, is_night: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (batch, seq_len, n_features)
            Covariate window.
        station_idx : Tensor (batch,)
            Station index for embedding lookup.
        clear_sky_ghi : Tensor (batch,)
            Clear-sky GHI at center timestep.
        is_night : Tensor (batch,)
            Nighttime flag at center timestep (1=night).

        Returns
        -------
        kt_pred : Tensor (batch, 1) -- predicted clearness index
        ghi_pred : Tensor (batch, 1) -- predicted radiation (physics-gated)
        """
        batch_size, seq_len, _ = x.shape

        # Station embedding: broadcast across sequence
        emb = self.station_embedding(station_idx)  # (batch, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, embed_dim)

        # Concatenate features + embedding
        x = torch.cat([x, emb], dim=-1)  # (batch, seq_len, n_features + embed_dim)

        # Layer normalization
        x = self.input_norm(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, 2*hidden_dim)

        # Extract CENTER hidden state (not last -- symmetric window)
        center_idx = self.half_window  # Index of center in the window
        center_hidden = lstm_out[:, center_idx, :]  # (batch, 2*hidden_dim)

        # Predict clearness index
        kt_pred = self.output_head(center_hidden)  # (batch, 1)

        # Reconstruct GHI: kt * clear_sky_ghi
        ghi_pred = kt_pred.squeeze(-1) * clear_sky_ghi  # (batch,)

        # Night gate: force GHI = 0 at night
        day_mask = 1.0 - is_night  # 0 at night, 1 during day
        ghi_pred = ghi_pred * day_mask  # (batch,)

        return kt_pred.squeeze(-1), ghi_pred
