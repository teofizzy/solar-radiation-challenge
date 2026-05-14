
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from src.config import MODEL_PARAMS

class DiagnosticEncoder(nn.Module):
    """Encodes static diagnostic descriptors into a latent vector."""
    def __init__(self, in_dim=5, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    """Projects patches of time-series into a latent space."""
    def __init__(self, in_channels, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(in_channels * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        P = self.patch_len
        
        # Reshape to patches: (B, T/P, P*F)
        x = x.unfold(1, P, P) # (B, T/P, F, P)
        x = x.permute(0, 1, 2, 3).contiguous().view(B, -1, F * P)
        
        x = self.proj(x)
        return self.norm(x)

class PhysicsInformedPatchTransformer(nn.Module):
    def __init__(self, n_features, n_stations,
                 d_model=128, nhead=8, num_layers=3, 
                 patch_len=16, stride=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(n_features, patch_len, d_model)
        
        # 2. Diagnostic Encoder
        self.diag_encoder = DiagnosticEncoder(in_dim=5, out_dim=32)
        
        # 3. Position Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, (MODEL_PARAMS['seq_len'] // patch_len), d_model))
        
        # 4. Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Station Memory Bank (for spatial cross-talk)
        self.station_memory = nn.Parameter(torch.randn(n_stations, d_model))
        
        # 6. Topographic Bias Matrix (Static)
        self.register_buffer('topo_bias', self._load_topo_bias(n_stations))
        self.topo_scale = nn.Parameter(torch.ones(1) * 0.1) # Learnable scale
        
        # 7. Fusion & Output Heads
        self.fusion = nn.Linear(d_model + 32, d_model)
        
        # Multi-task head: [delta_kt, raw_ghi_correction]
        self.head = nn.Linear(d_model, 2)
        

    def _load_topo_bias(self, n_stations):
        """Loads or pre-calculates topographic bias matrix."""
        # Load relative to this file for portability (Colab/Kaggle)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        bias_path = os.path.join(curr_dir, 'topographic_bias.pt')
        
        if os.path.exists(bias_path):
            print(f"[MODEL] Loading static topographic prior from {bias_path}")
            return torch.load(bias_path, weights_only=True)
            
        print("[MODEL] WARNING: Topographic bias not found. Using zero initialization.")
        return torch.zeros(n_stations, n_stations)

    def forward(self, x, station_idx, diag_vector,
                clear_sky_ghi, is_night, center_kt_landsaf):
        
        B = x.shape[0]
        
        # 1. Patching & Temporal Transformer
        x = self.patch_embed(x) # (B, n_patches, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x) # (B, n_patches, d_model)
        
        # 2. Multi-Token Pooling (Preserve advection phase)
        # Keep 4 central tokens instead of just 1
        n_p = x.shape[1]
        mid = n_p // 2
        z_temp = x[:, mid-2:mid+2, :] # (B, 4, d_model)
        
        # 3. Diagnostic Embedding
        z_diag = self.diag_encoder(diag_vector) # (B, 32)
        
        # 4. Spatial Attention (Advection Proxy)
        # query: (B, 4, d_model), key/value: (B, 40, d_model)
        query = z_temp 
        key = self.station_memory.unsqueeze(0).expand(B, -1, -1) # (B, 40, d_model)
        
        # Topographic Bias: scale is learnable
        bias_mask = (self.topo_bias[station_idx] * self.topo_scale).unsqueeze(1) # (B, 1, 40)
        
        d_k = self.d_model
        # scores: (B, 4, 40)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        scores = scores + bias_mask # Add Topographic Bias (broadcasts over 4 tokens)
        
        attn_weights = torch.softmax(scores, dim=-1)
        z_spatial = torch.matmul(attn_weights, key) # (B, 4, d_model)
        
        # Pool spatial tokens to 1
        z_spatial = z_spatial.mean(dim=1) # (B, d_model)
        
        # Fusion
        combined = torch.cat([z_spatial, z_diag], dim=-1)
        combined = F.gelu(self.fusion(combined))
        
        # Multi-task prediction: [delta_kt, raw_correction]
        out = self.head(combined)
        delta_kt_pred = out[:, 0] * 1.2
        raw_corr = out[:, 1] * 20.0 # Small correction in W/m2
        
        # Physics Reconstruct
        kt_pred = center_kt_landsaf + delta_kt_pred
        kt_pred = torch.clamp(kt_pred, 0.0, 1.4) # Relaxed upper bound (cloud enhancement)
        
        ghi_physics = kt_pred * clear_sky_ghi
        ghi_pred = ghi_physics + raw_corr # Final prediction with residual correction
        ghi_pred = ghi_pred * (1.0 - is_night)
        
        return delta_kt_pred, ghi_pred
