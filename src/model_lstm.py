"""
Physics-Informed BiLSTM for solar radiation reconstruction.
Predicts clearness index (kt) directly, then reconstructs GHI.

Architecture (V1 reverted + per-station bias):
    Input (batch, seq_len, n_features + embed_dim)
      -> LayerNorm
      -> BiLSTM (2 layers, bidirectional)
      -> Center timestep extraction
      -> Linear Head -> Sigmoid -> kt in [0, 1]
      -> GHI = kt * clear_sky_ghi * day_mask

Key design decisions (multi-AI consensus):
    1. Direct kt prediction (sigmoid-bounded) -- NOT delta_kt residual correction
    2. Per-station scalar bias embedding for TAHMO sensor drift (~2%/year)
    3. No CNN, no Transformer, no FiLM, no attention
    4. Hard night gate: GHI = 0 when is_night == 1

References:
    - NotebookLM [12]: CNN-BiLSTM RMSE=19.53, Attention-LSTM RMSE=30.64
    - Perplexity: "Direct kt prediction is cleaner optimization problem"
    - ChatGPT: "optimization geometry matters more than architecture sophistication"
"""

import torch
import torch.nn as nn

from src.config import HPARAMS


class PhysicsInformedBiLSTM(nn.Module):
    def __init__(self, n_features: int, n_stations: int,
                 hidden_dim: int = None, n_layers: int = None,
                 embed_dim: int = None, dropout: float = None):
        super().__init__()

        hidden_dim = hidden_dim or HPARAMS.get('hidden_dim', 160)
        n_layers = n_layers or HPARAMS.get('n_layers', 2)
        embed_dim = embed_dim or HPARAMS.get('station_embed_dim', 16)
        dropout = dropout or HPARAMS.get('dropout', 0.15)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.half_window = HPARAMS['half_window']

        # Station embedding: projects station index to learned representation
        self.station_embedding = nn.Embedding(n_stations, embed_dim)

        # Per-station scalar bias: corrects TAHMO sensor drift
        # Initialized to zero so it starts as identity correction
        self.station_bias = nn.Embedding(n_stations, 1)
        nn.init.constant_(self.station_bias.weight, 0.0)

        # Input normalization (stabilizes BiLSTM gradients)
        self.input_norm = nn.LayerNorm(n_features + embed_dim)

        # Bidirectional LSTM (core temporal model)
        self.bilstm = nn.LSTM(
            input_size=n_features + embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Output head: predict kt from center timestep hidden state
        # BiLSTM output dim = 2 * hidden_dim (forward + backward)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for RNNs, Xavier for linear layers."""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor,
                clear_sky_ghi: torch.Tensor, is_night: torch.Tensor):
        """
        Forward pass: predict kt directly, reconstruct GHI.

        Parameters
        ----------
        x : (batch, seq_len, n_features) -- covariate window
        station_idx : (batch,) -- station index for embedding
        clear_sky_ghi : (batch,) -- clear-sky GHI at center timestep
        is_night : (batch,) -- binary nighttime flag at center timestep

        Returns
        -------
        kt_pred : (batch,) -- predicted clearness index in [0, 1]
        ghi_pred : (batch,) -- predicted GHI (W/m2)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Station embedding: broadcast to sequence length
        emb = self.station_embedding(station_idx)  # (batch, embed_dim)
        emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, embed_dim)

        # 2. Concatenate features + station embedding
        x = torch.cat([x, emb_seq], dim=-1)  # (batch, seq_len, n_features + embed_dim)
        x = self.input_norm(x)

        # 3. BiLSTM forward pass
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, 2*hidden_dim)

        # 4. Extract center timestep representation (symmetric window)
        center_idx = self.half_window
        h_center = lstm_out[:, center_idx, :]  # (batch, 2*hidden_dim)

        # 5. Predict kt (bounded [0, 1] via sigmoid)
        kt_logit = self.head(h_center).squeeze(-1)  # (batch,)
        kt_pred = torch.sigmoid(kt_logit)

        # 6. Apply per-station bias (additive correction to kt)
        # Bias is small (~0.02 for 2% drift), clamped to [-0.15, 0.15]
        bias = self.station_bias(station_idx).squeeze(-1)  # (batch,)
        kt_pred = kt_pred + bias
        
        # 7. Clamp kt to physical bounds [0, kt_max]
        kt_max = HPARAMS.get('kt_max', 1.05)
        kt_pred = torch.clamp(kt_pred, 0.0, kt_max)

        # 8. Reconstruct GHI
        ghi_pred = kt_pred * clear_sky_ghi

        # 9. Hard night gate: force GHI = 0 at night
        day_mask = (1.0 - is_night).float()
        ghi_pred = ghi_pred * day_mask

        return kt_pred, ghi_pred
