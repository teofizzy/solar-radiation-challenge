"""
Physics-Informed CNN-BiLSTM for solar radiation reconstruction.
Predicts residual transmissivity (Delta kt), then reconstructs kt and GHI.

Architecture:
    Input (batch, seq_len, n_features + embed_dim)
      -> LayerNorm
      -> Temporal 1D-CNN (Local Feature Extraction)
      -> BiLSTM (2 layers, bidirectional)
      -> Center-Query Additive Attention (Temporal Focus)
      -> Global Head (Predicts Delta kt baseline)
      + Station Bias Head (Predicts pyranometer drift/bias)
      -> Delta kt = Global + Station Bias
      -> kt_pred = center_kt_landsaf + Delta kt
      -> GHI = kt_pred * clear_sky_ghi
"""

import torch
import torch.nn as nn

from src.config import HPARAMS


class PhysicsInformedCNNBiLSTM(nn.Module):
    def __init__(self, n_features: int, n_stations: int,
                 hidden_dim: int = None, n_layers: int = None,
                 embed_dim: int = None, dropout: float = None,
                 nhead: int = None):
        super().__init__()

        hidden_dim = hidden_dim or HPARAMS.get('hidden_dim', 128)
        n_layers = n_layers or HPARAMS.get('n_layers', 2)
        embed_dim = embed_dim or HPARAMS.get('embed_dim', 32)
        dropout = dropout or HPARAMS.get('dropout', 0.2)
        nhead = nhead or HPARAMS.get('transformer_heads', 8)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.half_window = HPARAMS['half_window']

        # Station embedding
        self.station_embedding = nn.Embedding(n_stations, embed_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features + embed_dim)

        # 1D-CNN for transient feature extraction (3 layers)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features + embed_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(in_channels=hidden_dim // 2, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Transformer Sequence Fusion
        nhead = nhead or HPARAMS.get('transformer_heads', 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1, # Reduced to 1 since Transformer handles deep sequence processing
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # Center-Query Additive Attention
        self.attn_W_q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_v = nn.Linear(hidden_dim * 2, 1)

        # Global Head to predict Delta kt
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # bounds output to [-1, 1], we scale to [-1.2, 1.2] below
        )

        # Station Bias Head to predict pyranometer drift
        self.station_bias_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # dynamic bounding to prevent runaway corrections
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for RNNs, Xavier for others."""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor,
                clear_sky_ghi: torch.Tensor, is_night: torch.Tensor,
                center_kt_landsaf: torch.Tensor):
        
        batch_size, seq_len, _ = x.shape

        # Embed station
        emb = self.station_embedding(station_idx)  # (batch, embed_dim)
        
        # Station Bias
        station_bias = self.station_bias_head(emb).squeeze(-1) # (batch,)

        # Broadcast embedding to sequence
        emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, emb_seq], dim=-1)
        x = self.input_norm(x)

        # CNN expects (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2) # (batch, seq_len, hidden_dim)

        # Transformer
        x_trans = self.transformer(x_cnn) # (batch, seq_len, hidden_dim)

        # BiLSTM
        lstm_out, _ = self.bilstm(x_trans)  # (batch, seq_len, 2*hidden_dim)

        # Center-Query Attention
        center_idx = self.half_window
        query = lstm_out[:, center_idx, :].unsqueeze(1) # (batch, 1, 2*hidden_dim)
        
        # Energy
        energy = self.attn_v(torch.tanh(self.attn_W_q(query) + self.attn_W_k(lstm_out))) # (batch, seq_len, 1)
        attention = torch.softmax(energy, dim=1)
        context = torch.sum(attention * lstm_out, dim=1) # (batch, 2*hidden_dim)

        # Global Delta Kt (scaled to [-1.2, 1.2])
        global_delta_kt = self.global_head(context).squeeze(-1) * 1.2
        
        # Final Delta Kt = Global + Bias (Bias scaled to [-0.1, 0.1] to act as slight correction)
        delta_kt_pred = global_delta_kt + (station_bias * 0.1)

        # Reconstruct Kt
        kt_pred = center_kt_landsaf + delta_kt_pred
        
        # Force strict physical bounds [0, 1.2]
        kt_pred = torch.clamp(kt_pred, 0.0, 1.2)

        # Reconstruct GHI
        ghi_pred = kt_pred * clear_sky_ghi

        # Night gate: force GHI = 0 at night
        day_mask = 1.0 - is_night
        ghi_pred = ghi_pred * day_mask

        return delta_kt_pred, ghi_pred, station_bias
