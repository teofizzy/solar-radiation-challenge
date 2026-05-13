"""
Attention-enhanced Physics-Informed BiLSTM for solar radiation reconstruction.

Architecture upgrade: replaces center-state-only extraction with
Center-Query Additive Attention (Bahdanau-style), resolving
"representation collapse" that discards 47/48 of BiLSTM temporal info.

Design decisions (3-AI consensus: ChatGPT, Gemini, Perplexity):
  - Attention type: Additive (Bahdanau) -- stable for anisotropic RNN states
  - Fusion: CONCAT [context, center] (unanimous)
  - Attention dropout: 0.10 after softmax (unanimous)
  - Temperature: tau=2.0 on scores (prevents attention collapse at T=48)
  - LayerNorm: on fused output only (unanimous)
  - Output head: 4H -> H -> 1 with GELU + Dropout(0.15) (unanimous)
"""

import torch
import torch.nn as nn

from src.config import HPARAMS


class CenterQueryAttention(nn.Module):
    """
    Center-query additive (Bahdanau) attention module.

    Uses the BiLSTM hidden state at the center timestep as query,
    attends over the full sequence, and produces a context vector
    concatenated with the center state via skip connection.

    Parameters
    ----------
    hidden_dim : int
        BiLSTM hidden dimension per direction (total BiLSTM output = 2*hidden_dim).
    attn_dim : int
        Attention projection dimension.
    dropout : float
        Dropout rate on attention weights after softmax.
    temperature : float
        Score scaling factor (scores / temperature) to prevent collapse.
    """

    def __init__(self, hidden_dim: int = 192, attn_dim: int = 128,
                 dropout: float = 0.10, temperature: float = 2.0):
        super().__init__()

        input_dim = 2 * hidden_dim  # BiLSTM is bidirectional

        # Query and Key projections (no bias, following Bahdanau convention)
        self.W_q = nn.Linear(input_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attn_dim, bias=False)

        # Score vector (projects tanh(q + k) to scalar)
        self.V = nn.Linear(attn_dim, 1, bias=False)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

        # LayerNorm on fused output (unanimous consensus)
        self.norm = nn.LayerNorm(2 * input_dim)  # cat[context, center] = 4H

    def forward(self, lstm_out: torch.Tensor):
        """
        Parameters
        ----------
        lstm_out : Tensor (B, T, 2H)
            Full BiLSTM sequence output.

        Returns
        -------
        fused : Tensor (B, 4H)
            Concatenated [context, center] with LayerNorm.
        alpha : Tensor (B, T)
            Attention weights (for visualization/debugging).
        """
        B, T, D = lstm_out.shape
        center_idx = T // 2

        # Center state as query
        center = lstm_out[:, center_idx]        # (B, 2H)

        # Project query and keys
        q = self.W_q(center)                    # (B, A)
        k = self.W_k(lstm_out)                  # (B, T, A)

        # Additive scoring: V * tanh(q + k)
        q_broadcast = q.unsqueeze(1)            # (B, 1, A)
        scores = self.V(
            torch.tanh(q_broadcast + k)
        ).squeeze(-1)                           # (B, T)

        # Temperature scaling to prevent attention collapse
        scores = scores / self.temperature

        # Softmax + dropout
        alpha = torch.softmax(scores, dim=1)    # (B, T)
        alpha_dropped = self.dropout(alpha)

        # Weighted context vector
        context = torch.sum(
            lstm_out * alpha_dropped.unsqueeze(-1),
            dim=1
        )                                       # (B, 2H)

        # CONCAT fusion with skip connection (not replace, not add)
        fused = torch.cat([context, center], dim=-1)  # (B, 4H)

        # LayerNorm on fused output
        fused = self.norm(fused)

        return fused, alpha


class AttentionBiLSTM(nn.Module):
    """
    Physics-Informed BiLSTM with Center-Query Attention.

    Replaces the original center-state-only extraction with attention-based
    temporal aggregation, using 48/48 of the sequence information instead of 1/48.

    Architecture:
        Input (batch, seq_len, n_features + embed_dim)
          -> LayerNorm
          -> BiLSTM (2 layers, hidden_dim, bidirectional)
          -> CenterQueryAttention (additive, tau=2.0)
          -> Linear(4H, H) -> GELU -> Dropout -> Linear(H, 1) -> Sigmoid
          -> GHI = kt * clear_sky_ghi
          -> Night gate: GHI * (1 - is_night)

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
        Dropout rate for LSTM and output head.
    attn_dim : int
        Attention projection dimension.
    attn_dropout : float
        Attention weight dropout rate.
    attn_temperature : float
        Attention score scaling factor.
    """

    def __init__(self, n_features: int, n_stations: int,
                 hidden_dim: int = None, n_layers: int = None,
                 embed_dim: int = None, dropout: float = None,
                 attn_dim: int = None, attn_dropout: float = None,
                 attn_temperature: float = None):
        super().__init__()

        hidden_dim = hidden_dim or HPARAMS['hidden_dim']
        n_layers = n_layers or HPARAMS['n_layers']
        embed_dim = embed_dim or HPARAMS['embed_dim']
        dropout = dropout or HPARAMS['dropout']
        attn_dim = attn_dim or HPARAMS.get('attn_dim', 128)
        attn_dropout = attn_dropout or HPARAMS.get('attn_dropout', 0.10)
        attn_temperature = attn_temperature or HPARAMS.get('attn_temperature', 2.0)

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

        # Center-query attention (replaces center-state-only extraction)
        self.attention = CenterQueryAttention(
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=attn_dropout,
            temperature=attn_temperature,
        )

        # Output head: fused dim = 4 * hidden_dim (concat of context + center)
        fused_dim = 4 * hidden_dim
        self.output_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),    # 4H -> H
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),            # H -> 1
            nn.Sigmoid(),                        # kt in (0, 1)
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

        # Initialize attention projections
        for module in [self.attention.W_q, self.attention.W_k, self.attention.V]:
            nn.init.xavier_uniform_(module.weight)

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
        kt_pred : Tensor (batch,) -- predicted clearness index
        ghi_pred : Tensor (batch,) -- predicted radiation (physics-gated)
        """
        batch_size, seq_len, _ = x.shape

        # Station embedding: broadcast across sequence
        emb = self.station_embedding(station_idx)  # (batch, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate features + embedding
        x = torch.cat([x, emb], dim=-1)

        # Layer normalization
        x = self.input_norm(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, 2*hidden_dim)

        # Center-query attention (replaces center_hidden = lstm_out[:, center, :])
        fused, attn_weights = self.attention(lstm_out)  # (batch, 4*hidden_dim)

        # Predict clearness index
        kt_pred = self.output_head(fused)  # (batch, 1)

        # Reconstruct GHI: kt * clear_sky_ghi
        ghi_pred = kt_pred.squeeze(-1) * clear_sky_ghi

        # Night gate: force GHI = 0 at night
        day_mask = 1.0 - is_night
        ghi_pred = ghi_pred * day_mask

        return kt_pred.squeeze(-1), ghi_pred
