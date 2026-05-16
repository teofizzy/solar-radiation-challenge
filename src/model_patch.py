
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from src.config import HPARAMS

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


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for per-station calibration.
    
    Each station learns a multiplicative (gamma) and additive (beta) 
    modulation of the hidden representation. This replaces the unbounded
    raw_correction head with a structured, per-station calibration that
    is far more stable and interpretable.
    
    Reference: Perez et al., "FiLM: Visual Reasoning with a General 
    Conditioning Layer", AAAI 2018.
    """
    def __init__(self, n_stations, d_model):
        super().__init__()
        self.gamma = nn.Embedding(n_stations, d_model)
        self.beta = nn.Embedding(n_stations, d_model)
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
    
    def forward(self, x, station_idx):
        """
        Args:
            x: (B, d_model) hidden representation
            station_idx: (B,) station indices
        Returns:
            (B, d_model) modulated representation
        """
        g = self.gamma(station_idx)  # (B, d_model)
        b = self.beta(station_idx)   # (B, d_model)
        return g * x + b


class PhysicsInformedSZAGate(nn.Module):
    """Learnable smooth Solar Zenith Angle gate (PISSM-inspired).
    
    Instead of a hard binary gate (sza <= 90), this learns a smooth
    sigmoid transition that preserves gradient flow at dawn/dusk.
    
    Reference: arXiv:2604.11807, Eqs. 14-15 -- Physics-Informed Gating.
    The gate output is in [0, 1] via sigmoid, naturally killing nighttime
    predictions smoothly rather than with a step function.
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cos_zenith):
        """
        Args:
            cos_zenith: (B,) cosine of solar zenith angle. 
                        >0 = daytime, <=0 = nighttime.
        Returns:
            (B,) gate values in [0, 1]
        """
        return self.gate(cos_zenith.unsqueeze(-1)).squeeze(-1)


class PatchEmbedding(nn.Module):
    """Projects patches of time-series into a latent space."""
    def __init__(self, in_channels, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(in_channels * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        P = self.patch_len
        
        # Reshape to patches: (B, T/P, P*F)
        # Corrected unfold logic: (B, n_patches, F, P) -> (B, n_patches, P, F) -> (B, n_patches, P*F)
        x = x.unfold(1, P, self.stride) # (B, n_patches, F, P)
        x = x.transpose(2, 3).contiguous() # (B, n_patches, P, F)
        x = x.view(B, -1, P * F)
        
        x = self.proj(x)
        return self.norm(x)

class PhysicsInformedPatchTransformer(nn.Module):
    """Two-Stage Physics-Informed PatchTransformer for solar GHI reconstruction.
    
    Stage 1 (this module): Predicts delta_kt ONLY (clearness index correction).
    Reconstructs GHI via pure physics: GHI = clamp(kt_landsaf + delta_kt, 0, 1.05) * GHI_cs * g_sza
    
    Key architectural changes from v1:
    - REMOVED: raw_correction head (was unbounded additive bias dump)
    - ADDED: FiLM conditioning (per-station gamma*x + beta modulation)
    - ADDED: PISSM-inspired learnable SZA gate (smooth sigmoid instead of binary)
    - CHANGED: Single output head (d_model -> 1) instead of multi-task (d_model -> 2)
    - CHANGED: kt clamp tightened from [0, 1.4] to [0, 1.05] (physical kt rarely > 1.0)
    
    Stage 2 (separate LightGBM): Corrects structured residuals from station drift,
    aerosol bias, and hour-of-day effects. See src/stage2_lgbm.py.
    """
    def __init__(self, n_features, n_stations,
                 d_model=128, nhead=8, num_layers=3, 
                 patch_len=16, stride=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(n_features, patch_len, stride, d_model)
        
        # 2. Diagnostic Encoder
        self.diag_encoder = DiagnosticEncoder(in_dim=5, out_dim=32)
        
        # 3. Position Encoding (Learnable)
        n_patches = (HPARAMS['seq_len'] - patch_len) // stride + 1
        self.n_patches = n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        
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
        
        # 7. Logit Scale (Temperature) for Cosine Attention
        # Bounds cosine similarity [-1, 1] to a broader range before softmax.
        # Initialize at log(10) ~ 2.3
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))

        # 8. Atmospheric Gate for dynamic spatial attention
        self.atmos_gate = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 9. Fusion & Output
        self.fusion = nn.Linear(d_model + 32, d_model)
        
        # 10. FiLM Conditioning (per-station scale+shift, replaces raw_correction)
        self.film = FiLMConditioner(n_stations, d_model)
        
        # 11. Single-task head: delta_kt ONLY (removed dual-head raw_correction)
        self.head = nn.Linear(d_model, 1)
        
        # 12. Physics-Informed SZA Gate (PISSM-inspired, replaces binary mask)
        self.sza_gate = PhysicsInformedSZAGate(hidden_dim=16)
        

    def _load_topo_bias(self, n_stations):
        """Loads topographic bias and converts to log-space for attention logits.
        
        Raw bias is in [0, 1] (similarity). Attention logits live in [-5, +5].
        Log-transform maps [0,1] -> [-inf, 0], giving weak connections a strong
        negative penalty. Normalize to unit variance for stable attention temperature.
        """
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        bias_path = os.path.join(curr_dir, 'topographic_bias.pt')
        
        if os.path.exists(bias_path):
            raw = torch.load(bias_path, weights_only=True)
            # Log-transform: similarity -> log-space penalty. Clamp to 1e-3 to prevent -inf.
            log_bias = torch.log(torch.clamp(raw, min=1e-3))
            # Normalize to zero-mean, unit-variance for stable injection
            log_bias = (log_bias - log_bias.mean()) / (log_bias.std() + 1e-8)
            print(f"[MODEL] Loaded log-transformed topographic prior from {bias_path}")
            print(f"  Log-bias range: [{log_bias.min():.2f}, {log_bias.max():.2f}]")
            return log_bias
            
        print("[MODEL] WARNING: Topographic bias not found. Using zero initialization.")
        return torch.zeros(n_stations, n_stations)

    def forward(self, x, station_idx, diag_vector,
                clear_sky_ghi, cos_zenith, center_kt_landsaf, atmos_feats):
        """
        Forward pass. Predicts delta_kt and reconstructs GHI via pure physics.
        
        Args:
            x: (B, seq_len, n_features) -- input feature window
            station_idx: (B,) -- station indices
            diag_vector: (B, 5) -- diagnostic descriptors
            clear_sky_ghi: (B,) -- clear-sky GHI at center timestep
            cos_zenith: (B,) -- cosine of solar zenith at center timestep
            center_kt_landsaf: (B,) -- LandSAF clearness index at center
            atmos_feats: (B, 4) -- atmospheric features [wind_speed, wind_dir, tcwv, cloud_frac]
        
        Returns:
            delta_kt_pred: (B,) -- predicted clearness index correction
            ghi_pred: (B,) -- reconstructed GHI (physics + SZA gate)
        
        Note: 'is_night' parameter removed. SZA gate now handles night suppression
              via learned cos_zenith gating (PISSM-inspired).
        """
        B = x.shape[0]
        
        # 0. Defensive Sanitization (Anti-NaN)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        diag_vector = torch.nan_to_num(diag_vector, nan=0.0)
        
        # 1. Patching & Temporal Transformer
        x = self.patch_embed(x) # (B, n_patches, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x) # (B, n_patches, d_model)
        
        # 2. Multi-Token Pooling (Preserve advection phase)
        # Keep up to 4 central tokens. Dynamic clamp for stability across hparam variations.
        n_p = x.shape[1]
        mid = n_p // 2
        start_idx = max(0, mid - 2)
        end_idx = min(n_p, mid + 2)
        z_temp = x[:, start_idx:end_idx, :] # (B, tokens, d_model)
        
        # 3. Diagnostic Embedding
        z_diag = self.diag_encoder(diag_vector) # (B, 32)
        
        # 4. Atmospheric Modulation of Query
        # Add atmospheric condition to the query tokens
        atmos_embed = self.atmos_gate(atmos_feats).unsqueeze(1) # (B, 1, d_model)
        z_query = z_temp + atmos_embed
        
        # query: (B, tokens, d_model), key/value: (B, 40, d_model)
        # Normalizing query and key converts matmul to cosine similarity.
        query = F.normalize(z_query, dim=-1) 
        key = F.normalize(self.station_memory, dim=-1).unsqueeze(0).expand(B, -1, -1) 
        
        # scores: (B, tokens, 40)
        # Cosine similarity is in [-1, 1]. Scaling expands range for selective attention.
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale logits with learnable temperature
        logit_scale_exp = self.logit_scale.exp().clamp(max=100.0)
        scores = scores * logit_scale_exp
        
        # Topographic Bias: scale is learnable. (B, 1, 40)
        bias_mask = (self.topo_bias[station_idx] * self.topo_scale).unsqueeze(1)
        scores = scores + bias_mask
        
        # Logit clamping prevents softmax saturation and INF gradients in AMP
        scores = torch.clamp(scores, min=-20, max=20)
        
        attn_weights = torch.softmax(scores, dim=-1) # (B, tokens, 40)
        z_spatial = torch.matmul(attn_weights, key) # (B, tokens, d_model)
        
        # Pool spatial tokens to 1
        z_spatial = z_spatial.mean(dim=1) # (B, d_model)
        
        # Fusion
        combined = torch.cat([z_spatial, z_diag], dim=-1)
        combined = F.gelu(self.fusion(combined))
        
        # FiLM Conditioning: per-station scale+shift modulation
        combined = self.film(combined, station_idx)
        
        # Single-task prediction: delta_kt ONLY (no raw_correction)
        delta_kt_pred = self.head(combined).squeeze(-1)  # (B,)
        
        # Physics Reconstruction (Force FP32 to prevent overflow in clear-sky multiplication)
        with torch.amp.autocast('cuda', enabled=False):
            delta_kt_pred_f32 = delta_kt_pred.float()
            center_kt_landsaf_f32 = center_kt_landsaf.float()
            clear_sky_ghi_f32 = clear_sky_ghi.float()
            cos_zenith_f32 = cos_zenith.float()

            # Reconstruct kt: kt_pred = kt_landsaf + delta_kt
            kt_pred = center_kt_landsaf_f32 + delta_kt_pred_f32
            # Physical constraint: kt in [0, 1.05] (rarely exceeds 1.0 in clear conditions)
            kt_pred = torch.clamp(kt_pred, 0.0, 1.05)
            
            # Physics GHI: GHI = kt * GHI_cs
            ghi_physics = kt_pred * clear_sky_ghi_f32
            
            # PISSM-Inspired SZA Gate: smooth learned suppression
            # Replaces binary (1 - is_night) with differentiable sigmoid gate
            g_sza = self.sza_gate(cos_zenith_f32)  # (B,) in [0, 1]
            ghi_pred = ghi_physics * g_sza
            
            # Final physical clamp: non-negative GHI
            ghi_pred = torch.clamp(ghi_pred, min=0.0)
            
        return delta_kt_pred, ghi_pred
