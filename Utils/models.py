"""
Surrogate models for multi-material light-trapping inverse design.

Models:
    1. ForwardMLP              - MLP forward model with material embedding
    2. SpatialCNN              - 1D CNN constructing grating profile in forward pass
    3. TandemNetwork           - Deterministic inverse model
    4. GenerativeTandemNetwork - Conditional generative inverse model
    5. ContrastiveVAE          - VAE with margin loss for shared latent neighbourhood

Data shapes (from Scripts/generate_dataset.py):
    - params_x:      (N, N_harmonics, 2) — [amplitude, phase] per harmonic
    - h:             scalar bulk height [nm]
    - material_id:   int in {0, ..., N_materials-1}
    - absorptance:   (N, N_wavelengths) — 161 points, 300–1100 nm

Grating materials: {'Si': 0, 'TiO2': 1, 'Si3N4': 2}
Ag is reflector-only — not a valid grating material.

References:
    - Snake activation: Ziyin et al. 2020, "Neural Networks Fail to Learn
      Periodic Functions and How to Fix It"
    - Gumbel-Softmax: Jang et al. 2017, "Categorical Reparameterization
      with Gumbel-Softmax"
    - Tandem network: Liu et al. 2018, "Training Deep Neural Networks for
      the Inverse Design of Nanophotonic Structures"
    - SIREN: Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions"
      NeurIPS 2020.
    - Uncertainty Weighting: Kendall et al. 2018, "Multi-Task Learning Using Uncertainty to Weigh
      Losses for Scene Geometry and Semantics"
"""
import sys
import math
from typing import Optional, Literal, Sequence, Dict

from tqdm import tqdm

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ag is reflector-only and NOT available as a grating material.
MATERIAL_LIBRARY: Dict[str, int] = {
    "Si": 0, "TiO2": 1, "Si3N4": 2,
    "Si_Ag": 3, "TiO2_Ag": 4, "Si3N4_Ag": 5
}
N_MATERIALS: int = len(MATERIAL_LIBRARY)


class Snake(nn.Module):
    """Snake activation: x + (1/a) sin²(ax). Learnable per-channel frequency."""

    def __init__(self, in_features: int, a_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((in_features,), a_init))

    def forward(self, x):
        a = self.a
        if x.dim() == 3:
            a = a.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            a = a.unsqueeze(0)
        return x + (1.0 / a) * torch.sin(a * x) ** 2



def _make_activation(name: str, dim: int) -> nn.Module:
    if name == "snake":
        return Snake(dim)
    elif name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


def _embed_material(material_id: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    """Handles both int64 IDs and float one-hot (for differentiable Gumbel path)."""
    if material_id.dtype in (torch.float32, torch.float64):
        return material_id @ embedding.weight
    return embedding(material_id)



def build_profile(geometry, n_harmonics, nx=128, grating_period=1000.0, r_grid=None, harmonic_idx=None):
    n_fourier = n_harmonics * 2
    params_x = geometry[:, :n_fourier].view(-1, n_harmonics, 2)
    amps = params_x[:, :, 0]
    phases = params_x[:, :, 1]
    grating_height = 2.0 * amps.sum(dim=1, keepdim=True) + 1e-9

    device = geometry.device
    if r_grid is None:
        r_grid = torch.linspace(0, grating_period, nx + 1, device=device)[:-1]
    if harmonic_idx is None:
        harmonic_idx = torch.arange(1, n_harmonics + 1, dtype=torch.float32, device=device)

    n = harmonic_idx[None, :, None]
    r = r_grid[None, None, :]
    arg = 2.0 * math.pi * n * r / grating_period - phases[:, :, None]
    cosines = amps[:, :, None] * torch.cos(arg)
    profile = grating_height[:, :, None] / 2.0 + cosines.sum(dim=1, keepdim=True)
    
    h = geometry[:, n_fourier:n_fourier+1]
    inc_ang = geometry[:, n_fourier+1:n_fourier+2]

    return profile.squeeze(1), h, inc_ang


class SkipLinear(nn.Module):
    def __init__(self, in_features, out_features, activation="gelu", norm="layer", dropout=0.05):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.Identity()
        self.act = _make_activation(activation, out_features)
        self.dropout = nn.Dropout(dropout)
        self.has_skip = (in_features == out_features)
        
    def forward(self, x):
        h = self.dropout(self.act(self.norm(self.linear(x))))
        return x + h if self.has_skip else h

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                limit = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-limit, limit)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.05, downsample=False, activation="gelu"):
        super().__init__()
        stride = 2 if downsample else 1
        pad_mode = "zeros" if downsample else "circular"
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=pad_mode)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.act1 = _make_activation(activation, out_ch)
        self.drop1 = nn.Dropout1d(dropout)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2, padding_mode="circular")
        self.norm2 = nn.BatchNorm1d(out_ch)
        
        if in_ch != out_ch or downsample:
            layers = []
            if downsample:
                # Use AvgPool to downsample without dropping pixels and without learning new spatial mixings
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True))
                
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_ch)
            ])
            self.skip = nn.Sequential(*layers)
        else:
            self.skip = nn.Identity()
            
        self.act2 = _make_activation(activation, out_ch)
        self.drop2 = nn.Dropout1d(dropout)

    def forward(self, x):
        res = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.drop2(self.act2(x + res))


class SIREN(nn.Module):
    """
    A Hybrid 'Conditioned Implicit Neural Representation'.
    1. A CNN encodes the global 1D physical parameters into a latent vector.
    2. A SIREN decoder takes (latent_vector, normalized_wavelength) -> (P_abs, S_abs)
    This breaks the Spectral Bias trap, learning both global mappings and infinite-resolution sharp peaks.
    """
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=322, n_materials=3, embed_dim=8, 
                 conv_channels=(32, 64, 128), kernel_size=7, dropout=0.0, siren_hidden=(256, 256, 256), latent_dim=64, omega_0=30.0, **kwargs):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.seq_len = n_wavelengths // 2
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        
        in_ch = 1 + 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        # 1. CNN Encoder (matching SkipCNN)
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout, downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.encoder_cnn = nn.Sequential(*conv_layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + 1 + embed_dim + 1, nx)
            spatial_dim = self.encoder_cnn(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_dim        
        # Project CNN output to latent vector
        self.encoder_proj = nn.Sequential(
            SkipLinear(fc_in, 256, activation="gelu", norm="layer", dropout=dropout),
            nn.Linear(256, latent_dim)
        )
        
        # 2. SIREN Decoder
        siren_in_dim = latent_dim + 1 # latent_vector + 1D normalized wavelength
        siren_layers = []
        for i, h_dim in enumerate(siren_hidden):
            siren_layers.append(SineLayer(siren_in_dim, h_dim, is_first=(i==0), omega_0=omega_0))
            siren_in_dim = h_dim
            
        self.siren_decoder = nn.Sequential(*siren_layers)
        self.head = nn.Linear(siren_in_dim, 2)

    def forward(self, geometry, material_id, wls=None, profile=None, h=None, inc_ang=None):
        B = geometry.shape[0] if geometry is not None else profile.shape[0]
        if profile is None:
            profile, h, inc_ang = build_profile(geometry, self.n_harmonics, self.nx)
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        L = profile.shape[1]
        h_spatial = h.unsqueeze(2).expand(B, 1, L)
        mat_spatial = mat_embed.unsqueeze(2).expand(B, -1, L)
        inc_ang_spatial = inc_ang.unsqueeze(2).expand(B, 1, L)
        
        x_list = [profile.unsqueeze(1), h_spatial, mat_spatial, inc_ang_spatial]
        x = torch.cat(x_list, dim=1)
        
        x = self.input_norm(x)
        x = self.encoder_cnn(x)
        x = x.view(B, -1)
        latent = self.encoder_proj(x)
        
        if wls is None:
            wls_eval = torch.linspace(300, 1100, self.seq_len, device=geometry.device) + 1e-3
            W = self.seq_len
            wls_expanded = wls_eval.view(1, W, 1).expand(B, -1, -1)
            return_flat = True
        else:
            wls_eval = wls
            if wls.dim() == 1:
                W = wls.shape[0]
                wls_expanded = wls.view(1, W, 1).expand(B, -1, -1)
            else:
                W = wls.shape[1]
                wls_expanded = wls.unsqueeze(-1)
            return_flat = False
            
        # Normalize to [-1, 1] across [300, 1100]
        wls_norm = (wls_expanded - 700.0) / 400.0
        
        latent_expanded = latent.unsqueeze(1).expand(B, W, -1)
        siren_in = torch.cat([latent_expanded, wls_norm], dim=-1)
        out = torch.sigmoid(self.head(self.siren_decoder(siren_in)))
        
        if return_flat:
            p_pol = out[..., 0]
            s_pol = out[..., 1]
            return torch.cat([p_pol, s_pol], dim=1)
        return out

class ForwardMLP(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, hidden_dims=(512, 512, 512), norm="layer", dropout=0.0, grating_period=1000.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.grating_period = grating_period
        self.material_embedding = nn.Embedding(n_materials, embed_dim)

        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_dim = nx + 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_dim)

        layers = []
        for h_dim in hidden_dims:
            layers.append(SkipLinear(in_dim, h_dim, activation="gelu", norm=norm, dropout=dropout))
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_wavelengths)

    def forward(self, geometry=None, material_id=None, profile=None, h=None, inc_ang=None):
        B = geometry.shape[0] if geometry is not None else profile.shape[0]
        if profile is None:
            profile, h, inc_ang = build_profile(geometry, self.n_harmonics, self.nx, self.grating_period, self.r_grid, self.harmonic_idx)
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        x_list = [profile, h, mat_embed, inc_ang]
        x = torch.cat(x_list, dim=-1)
        x = self.input_norm(x)
        return torch.sigmoid(self.head(self.trunk(x)))

class SkipCNN(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, grating_period=1000.0, conv_channels=(32, 64, 128, 64), kernel_size=7, fc_dims=(512, 256), dropout=0.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.grating_period = grating_period
        
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_ch = 1 + embed_dim + 1 # abs_profile + mat_embed + inc_ang
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout, downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1 + embed_dim + 1, nx)
            spatial_nx = self.conv_backbone(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_nx        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        fc_layers.append(nn.Sigmoid())
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, geometry=None, material_id=None, profile=None, h=None, inc_ang=None):
        B = geometry.shape[0] if geometry is not None else profile.shape[0]
        if profile is None:
            profile, h, inc_ang = build_profile(
                geometry, self.n_harmonics, self.nx, 
                self.grating_period, self.r_grid, self.harmonic_idx
            )
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        B, L = profile.shape
        abs_profile = profile + h 
        
        x_prof = abs_profile.view(B, 1, L)
        x_mat = mat_embed.view(B, -1, 1).expand(B, -1, L)
        x_inc = inc_ang.view(B, 1, 1).expand(B, 1, L)
        
        x = torch.cat([x_prof, x_mat, x_inc], dim=1)
        x = self.input_norm(x)
        x = self.conv_backbone(x)
        x = x.view(B, -1)
        return self.fc_head(x)

class SpatialCNN(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, grating_period=1000.0, conv_channels=(32, 64, 128, 64), kernel_size=7, fc_dims=(512, 256), dropout=0.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.grating_period = grating_period
        
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_ch = 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout1d(dropout),
            ]
            if i < len(conv_channels) - 1:
                conv_layers += [
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(out_ch), nn.GELU(),
                ]
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1 + embed_dim + 1, nx)
            spatial_nx = self.conv_backbone(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_nx        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        fc_layers.append(nn.Sigmoid())
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, geometry=None, material_id=None, profile=None, h=None, inc_ang=None):
        B = geometry.shape[0] if geometry is not None else profile.shape[0]
        if profile is None:
            profile, h, inc_ang = build_profile(
                geometry, self.n_harmonics, self.nx, 
                self.grating_period, self.r_grid, self.harmonic_idx
            )
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        B, L = profile.shape
        abs_profile = profile + h 
        
        x_prof = abs_profile.view(B, 1, L)
        x_mat = mat_embed.view(B, -1, 1).expand(B, -1, L)
        x_inc = inc_ang.view(B, 1, 1).expand(B, 1, L)
        
        x = torch.cat([x_prof, x_mat, x_inc], dim=1)
        x = self.input_norm(x)
        x = self.conv_backbone(x)
        x = x.view(B, -1)
        return self.fc_head(x)



class TransformerForward(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=322, n_materials=3, 
                 embed_dim=8, d_model=128, nhead=4, dim_feedforward=512, num_layers=3, dropout=0.0, grating_period=1000.0,
                 patch_size=8):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.grating_period = grating_period
        self.patch_size = patch_size
        self.num_patches = nx // patch_size
        
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        
        # Patch projector (Extracts non-overlapping patches from the 1D profile)
        self.patch_proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Global Token projectors
        self.h_proj = nn.Linear(1, d_model)
        self.inc_ang_proj = nn.Linear(1, d_model)
        self.mat_proj = nn.Linear(embed_dim, d_model)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional Encoding (learnable)
        self.seq_len = self.num_patches + 4  # e.g., 16 patches + h + inc + mat + cls = 20
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_wavelengths),
            nn.Sigmoid()
        )

    def forward(self, geometry=None, material_id=None, wls=None, profile=None, h=None, inc_ang=None):
        B = geometry.shape[0] if geometry is not None else profile.shape[0]
        if profile is None:
            profile, h, inc_ang = build_profile(
                geometry, self.n_harmonics, self.nx, 
                self.grating_period, self.r_grid, self.harmonic_idx
            )
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        B, L = profile.shape
        abs_profile = profile + h 
        
        # Extract and project patches
        x_prof = abs_profile.view(B, 1, L)
        patch_tokens = self.patch_proj(x_prof)         # (B, d_model, num_patches)
        patch_tokens = patch_tokens.transpose(1, 2)    # (B, num_patches, d_model)
        
        # Project global tokens
        h_token = self.h_proj(h.view(B, 1)).unsqueeze(1) # (B, 1, d_model)
        inc_token = self.inc_ang_proj(inc_ang.view(B, 1)).unsqueeze(1) # (B, 1, d_model)
        mat_token = self.mat_proj(mat_embed.view(B, -1)).unsqueeze(1) # (B, 1, d_model)
        
        # Expand [CLS] token for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        
        # Sequence: [CLS, h, inc, mat, ...patches...]
        x = torch.cat([cls_tokens, h_token, inc_token, mat_token, patch_tokens], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Pooling: Extract ONLY the [CLS] token's output representation (index 0)
        cls_out = x[:, 0, :]
        return self.head(cls_out)

def generate_synthetic_targets(B: int, n_wavelengths: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic ideal targets with a physically motivated distribution.

    Distribution:
      - 1/3  Gaussian peaks (random centre, width)            – spectral selectivity
      - 1/3  Single rectangular band (random centre, width)   – band-specific targets
      - 1/6  Multi-band (2–3 random rectangular bands, union) – multi-resonance capability
      - 1/6  Broadband all-ones                               – hardest, most solar-relevant goal

    Returns:
        targets  – (B, n_wavelengths)  values in [0, 1]
        weights  – (B, n_wavelengths)  1.0 in active regions, 0.1 elsewhere
    """
    wl_len = n_wavelengths // 2
    wls = torch.linspace(300, 1100, wl_len, device=device).unsqueeze(0).expand(B, -1)  # (B, wl_len)

    targets = torch.zeros(B, wl_len, device=device)
    weights = torch.full((B, wl_len), 0.1, device=device)

    # Assign each sample to one of four modes
    # Thresholds: 0 – 1/3 Gauss | 1/3 – 2/3 single | 2/3 – 5/6 multi | 5/6 – 1 broadband
    rand_mode = torch.rand(B, device=device)
    mask_gauss     = rand_mode < 1/3
    mask_single    = (rand_mode >= 1/3) & (rand_mode < 2/3)
    mask_multi     = (rand_mode >= 2/3) & (rand_mode < 5/6)
    mask_broadband = rand_mode >= 5/6

    # --- Gaussians ---
    mu    = torch.rand(B, 1, device=device) * 800 + 300          # centre in [300, 1100]
    sigma = torch.rand(B, 1, device=device) * 150 + 50           # width   in [50,  200]
    t_gauss = torch.exp(-0.5 * ((wls - mu) / sigma) ** 2)
    w_gauss = torch.where(torch.abs(wls - mu) <= sigma, 1.0, 0.1)
    targets[mask_gauss] = t_gauss[mask_gauss]
    weights[mask_gauss] = w_gauss[mask_gauss]

    # --- Single rectangular band ---
    c_s     = torch.rand(B, 1, device=device) * 800 + 300        # centre in [300, 1100]
    w_s     = torch.rand(B, 1, device=device) * 300 + 50         # width  in [50,  350]
    in_band = (wls >= c_s - w_s / 2) & (wls <= c_s + w_s / 2)
    t_single = in_band.float()
    w_single = torch.where(in_band, 1.0, 0.1)
    targets[mask_single] = t_single[mask_single]
    weights[mask_single] = w_single[mask_single]

    # --- Multi-band (2–3 random rectangular bands, union) ---
    n_bands_options = torch.randint(2, 4, (B,), device=device)   # 2 or 3 bands per sample
    t_multi = torch.zeros(B, wl_len, device=device)
    w_multi = torch.full((B, wl_len), 0.1, device=device)
    max_bands = 3
    for k in range(max_bands):
        active = n_bands_options > k                              # (B,)
        c_k = torch.rand(B, 1, device=device) * 700 + 350        # keep bands well inside [300,1100]
        w_k = torch.rand(B, 1, device=device) * 150 + 30         # narrower bands: [30, 180] nm
        in_k = (wls >= c_k - w_k / 2) & (wls <= c_k + w_k / 2)
        t_multi = torch.where(active.unsqueeze(1) & in_k, torch.ones_like(t_multi), t_multi)
        w_multi = torch.where(active.unsqueeze(1) & in_k, torch.ones_like(w_multi), w_multi)
    targets[mask_multi] = t_multi[mask_multi]
    weights[mask_multi] = w_multi[mask_multi]

    # --- Broadband all-ones (uniform absorption target across full spectrum) ---
    # Weight mask is uniformly 1.0 so the loss penalises all wavelengths equally.
    targets[mask_broadband] = 1.0
    weights[mask_broadband] = 1.0

    # Duplicate for s-pol (same target for both polarisations)
    targets = targets.repeat(1, 2)
    weights = weights.repeat(1, 2)

    return targets, weights



class InverseDecoder(nn.Module):
    """Maps absorptance curve (+ optional noise z) → normalized geometry [0,1] + Gumbel material."""

    def __init__(
        self,
        n_wavelengths: int = 161,
        n_geometry: int = 12,
        n_materials: int = 3,
        latent_dim: int = 0,
        geo_min: Optional[torch.Tensor] = None,
        geo_max: Optional[torch.Tensor] = None,
        conv_channels: Sequence[int] = (32, 64, 128, 64),
        kernel_size: int = 7,
        fc_dims: Sequence[int] = (256, 256),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_geometry = n_geometry
        self.n_materials = n_materials
        self.latent_dim = latent_dim
        self.seq_len = n_wavelengths // 2

        if geo_min is None: geo_min = torch.zeros(n_geometry)
        if geo_max is None: geo_max = torch.ones(n_geometry)
        self.register_buffer("geo_min", geo_min.view(1, -1))
        self.register_buffer("geo_max", geo_max.view(1, -1))

        in_ch = 2
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout, downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.seq_len)
            spatial_dim = self.conv_backbone(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_dim + latent_dim        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        self.fc_head = nn.Sequential(*fc_layers)

        N = (n_geometry - 2) // 2
        self.geometry_head = nn.Linear(fc_in, n_geometry + N)
        self.material_head = nn.Linear(fc_in, n_materials)

    def forward(
        self, target_curve: torch.Tensor, z: Optional[torch.Tensor] = None,
        tau: float = 1.0, hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred_geometry_physical, material_onehot, material_logits)."""
        B = target_curve.shape[0]
        
        # Reshape [B, 322] -> [B, 2, 161]
        x = target_curve.view(B, 2, self.seq_len)
        x = self.input_norm(x)
        x = self.conv_backbone(x)
        x = x.view(B, -1) # Flatten
        
        if z is not None:
            x = torch.cat([x, z], dim=-1)
            
        h = self.fc_head(x)
        
        raw_geo = self.geometry_head(h)
        N = (self.n_geometry - 2) // 2
        amps_norm = torch.sigmoid(raw_geo[:, :N])
        xy = raw_geo[:, N:3*N].view(B, N, 2)
        h_inc_norm = torch.sigmoid(raw_geo[:, 3*N:])
        
        phases = torch.atan2(xy[:, :, 1], xy[:, :, 0]) + math.pi
        amps = amps_norm * (self.geo_max[:, 0:2*N:2] - self.geo_min[:, 0:2*N:2]) + self.geo_min[:, 0:2*N:2]
        
        pred_geometry = torch.empty(B, self.n_geometry, device=raw_geo.device)
        pred_geometry[:, 0:2*N:2] = amps
        pred_geometry[:, 1:2*N:2] = phases
        pred_geometry[:, 2*N:] = h_inc_norm * (self.geo_max[:, 2*N:] - self.geo_min[:, 2*N:]) + self.geo_min[:, 2*N:]
        
        material_logits = self.material_head(h)
        material_onehot = F.gumbel_softmax(material_logits, tau=tau, hard=hard)
        return pred_geometry, material_onehot, material_logits


class TandemNetwork(nn.Module):
    """Inverse design via frozen forward surrogate.

    Loss: MSE(forward(predicted_geo, predicted_mat), target_curve).
    Only the InverseDecoder is trained; the forward model is frozen.
    """

    def __init__(self, inverse_decoder: InverseDecoder, forward_model: nn.Module):
        super().__init__()
        self.inverse_decoder = inverse_decoder
        self.forward_model = forward_model
        for param in self.forward_model.parameters():
            param.requires_grad = False
        self.forward_model.eval()

    def forward(self, target_curve: torch.Tensor, tau: float = 1.0) -> dict[str, torch.Tensor]:
        pred_geo, mat_oh, mat_logits = self.inverse_decoder(target_curve, tau=tau, hard=True)
        predicted_curve = self.forward_model(geometry=pred_geo, material_id=mat_oh)
        return {"predicted_curve": predicted_curve, "pred_geometry": pred_geo,
                "material_onehot": mat_oh, "material_logits": mat_logits}

    def train(self, mode: bool = True):
        super().train(mode)
        self.forward_model.eval()
        return self


class GenerativeTandemNetwork(nn.Module):
    """Tandem with conditional noise z for diverse inverse solutions.

    Different z ~ N(0,I) → different valid geometries for the same target.
    Loss: MSE(predicted_curve, target_curve).
    """

    def __init__(self, inverse_decoder: InverseDecoder, forward_model: nn.Module, latent_dim: int = 32):
        super().__init__()
        assert inverse_decoder.latent_dim == latent_dim
        self.inverse_decoder = inverse_decoder
        self.latent_dim = latent_dim
        self.forward_model = forward_model
        for param in self.forward_model.parameters():
            param.requires_grad = False
        self.forward_model.eval()

    def forward(
        self, target_curve: torch.Tensor, z: Optional[torch.Tensor] = None, tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        B = target_curve.shape[0]
        if z is None:
            z = torch.randn(B, self.latent_dim, device=target_curve.device)
        pred_geo, mat_oh, mat_logits = self.inverse_decoder(target_curve, z=z, tau=tau, hard=True)
        predicted_curve = self.forward_model(geometry=pred_geo, material_id=mat_oh)
        return {"predicted_curve": predicted_curve, "pred_geometry": pred_geo,
                "material_onehot": mat_oh, "material_logits": mat_logits, "z": z}

    @torch.no_grad()
    def sample_diverse_designs(
        self, target_curve: torch.Tensor, n_samples: int = 16, tau: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Generate n_samples diverse proposals for a single target curve."""
        if target_curve.dim() == 1:
            target_curve = target_curve.unsqueeze(0)
        target_repeated = target_curve.expand(n_samples, -1)
        z = torch.randn(n_samples, self.latent_dim, device=target_curve.device)
        return self.forward(target_repeated, z=z, tau=tau)

    def train(self, mode: bool = True):
        super().train(mode)
        self.forward_model.eval()
        return self


# --- Contrastive VAE components ---

class GeometryEncoder(nn.Module):
    """VAE encoder: geometry + material → (μ_x, log σ²_x)."""

    def __init__(
        self, n_continuous: int = 12, n_materials: int = N_MATERIALS, embed_dim: int = 8,
        latent_dim: int = 64, conv_channels: Sequence[int] = (32, 64, 64), kernel_size: int = 7,
        fc_dims: Sequence[int] = (256,), dropout: float = 0.05,
        n_harmonics: int = 5, nx: int = 128, grating_period: float = 1000.0,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.grating_period = grating_period
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_ch = 1 + 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout, downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1 + 1 + embed_dim + 1, nx)
            spatial_dim = self.conv_backbone(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_dim        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        self.fc_head = nn.Sequential(*fc_layers)
        
        self.fc_mu = nn.Linear(fc_in, latent_dim)
        self.fc_logvar = nn.Linear(fc_in, latent_dim)

    def forward(self, geometry: torch.Tensor, material_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        profile, h, inc_ang = build_profile(geometry, self.n_harmonics, self.nx, self.grating_period, self.r_grid, self.harmonic_idx)
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        B, L = profile.shape
        h_spatial = h.unsqueeze(2).expand(B, 1, L)
        mat_spatial = mat_embed.unsqueeze(2).expand(B, -1, L)
        inc_ang_spatial = inc_ang.unsqueeze(2).expand(B, 1, L)
        
        x_list = [profile.unsqueeze(1), h_spatial, mat_spatial, inc_ang_spatial]
        x = torch.cat(x_list, dim=1)
        x = self.input_norm(x)
        x = self.conv_backbone(x)
        x = x.view(B, -1)
        
        feat = self.fc_head(x)
        return self.fc_mu(feat), self.fc_logvar(feat)


class GeometryDecoder(nn.Module):
    """VAE decoder: Z_x → normalized geometry [0,1] + Gumbel material."""

    def __init__(
        self, latent_dim: int = 64, n_geometry: int = 11, n_materials: int = N_MATERIALS,
        geo_min: Optional[torch.Tensor] = None, geo_max: Optional[torch.Tensor] = None,
        hidden_dims: Sequence[int] = (256, 256), dropout: float = 0.05,
    ):
        super().__init__()
        self.n_geometry = n_geometry
        self.n_materials = n_materials

        if geo_min is None: geo_min = torch.zeros(n_geometry)
        if geo_max is None: geo_max = torch.ones(n_geometry)
        self.register_buffer("geo_min", geo_min.view(1, -1))
        self.register_buffer("geo_max", geo_max.view(1, -1))

        in_dim = latent_dim
        layers: list[nn.Module] = []
        for h_dim in hidden_dims:
            layers.append(SkipLinear(in_dim, h_dim, activation="gelu", norm="layer", dropout=dropout))
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        N = (n_geometry - 2) // 2
        self.geometry_head = nn.Linear(in_dim, n_geometry + N)
        self.material_head = nn.Linear(in_dim, n_materials)

    def forward(self, z: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """Returns (recon_geometry_physical, material_onehot, material_logits)."""
        h = self.trunk(z)
        raw_geo = self.geometry_head(h)
        B = raw_geo.shape[0]
        N = (self.n_geometry - 2) // 2
        amps_norm = torch.sigmoid(raw_geo[:, :N])
        xy = raw_geo[:, N:3*N].view(B, N, 2)
        h_inc_norm = torch.sigmoid(raw_geo[:, 3*N:])
        
        phases = torch.atan2(xy[:, :, 1], xy[:, :, 0]) + math.pi
        amps = amps_norm * (self.geo_max[:, 0:2*N:2] - self.geo_min[:, 0:2*N:2]) + self.geo_min[:, 0:2*N:2]
        
        recon_geometry = torch.empty(B, self.n_geometry, device=raw_geo.device)
        recon_geometry[:, 0:2*N:2] = amps
        recon_geometry[:, 1:2*N:2] = phases
        recon_geometry[:, 2*N:] = h_inc_norm * (self.geo_max[:, 2*N:] - self.geo_min[:, 2*N:]) + self.geo_min[:, 2*N:]
        material_logits = self.material_head(h)
        material_onehot = F.gumbel_softmax(material_logits, tau=tau, hard=hard)
        return recon_geometry, material_onehot, material_logits

class SpectrumEncoder(nn.Module):
    """Deterministic encoder: target curve → latent center Z_y."""

    def __init__(
        self, n_wavelengths: int = 161, latent_dim: int = 64,
        conv_channels: Sequence[int] = (32, 64, 128), kernel_size: int = 7,
        fc_dims: Sequence[int] = (256,), dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = n_wavelengths // 2
        
        in_ch = 2
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout, downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.seq_len)
            spatial_dim = self.conv_backbone(dummy).shape[-1]
        fc_in = conv_channels[-1] * spatial_dim        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
            
        self.fc_head = nn.Sequential(*fc_layers)
        self.latent_head = nn.Linear(fc_in, latent_dim)

    def forward(self, target_curve: torch.Tensor) -> torch.Tensor:
        B = target_curve.shape[0]
        x = target_curve.view(B, 2, self.seq_len)
        x = self.input_norm(x)
        x = self.conv_backbone(x)
        x = x.view(B, -1)
        h = self.fc_head(x)
        return self.latent_head(h)


class ContrastiveVAE(nn.Module):
    """Contrastive VAE with margin loss for shared geometry–spectrum latent space.

    Avoids catastrophic collapse of standard Joint AEs by using a hinge loss
    that only penalises Z_x falling outside a margin_radius of Z_y.

    Loss = MSE_recon + CE_material + β·KL + γ·margin

    Inference: encode target → Z_y, sample within B(Z_y, r), decode to geometry.
    """

    def __init__(
        self, geometry_encoder: GeometryEncoder, geometry_decoder: GeometryDecoder,
        spectrum_encoder: SpectrumEncoder, margin_radius: float = 1.0,
        beta: float = 1e-3, gamma: float = 1.0,
    ):
        super().__init__()
        self.geometry_encoder = geometry_encoder
        self.geometry_decoder = geometry_decoder
        self.spectrum_encoder = spectrum_encoder
        self.margin_radius = margin_radius
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Z_x = μ + σ * ε, ε ~ N(0,I)."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I))."""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    @staticmethod
    def margin_loss(z_x: torch.Tensor, z_y: torch.Tensor, margin_radius: float) -> torch.Tensor:
        """Hinge loss: mean(ReLU(||Z_x - Z_y||₂ - r))."""
        return torch.mean(F.relu(torch.norm(z_x - z_y, p=2, dim=-1) - margin_radius))

    def forward(
        self, geometry: torch.Tensor, material_id: torch.Tensor,
        target_curve: torch.Tensor, tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mu_x, logvar_x = self.geometry_encoder(geometry, material_id)
        z_x = self.reparameterise(mu_x, logvar_x)
        recon_geo, recon_mat_oh, recon_mat_logits = self.geometry_decoder(z_x, tau=tau, hard=True)
        z_y = self.spectrum_encoder(target_curve)
        return {"z_x": z_x, "z_y": z_y, "mu_x": mu_x, "logvar_x": logvar_x,
                "recon_geometry": recon_geo, "recon_material_onehot": recon_mat_oh,
                "recon_material_logits": recon_mat_logits}

    def compute_loss(
        self, out: dict[str, torch.Tensor], geometry: torch.Tensor, material_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        geo_min = self.geometry_decoder.geo_min
        geo_max = self.geometry_decoder.geo_max
        
        pred_norm = (out["recon_geometry"] - geo_min) / (geo_max - geo_min + 1e-9)
        targ_norm = (geometry - geo_min) / (geo_max - geo_min + 1e-9)
        
        loss_recon = F.huber_loss(pred_norm, targ_norm, delta=0.01)
        loss_mat_ce = F.cross_entropy(out["recon_material_logits"], material_id)
        loss_kl = self.kl_divergence(out["mu_x"], out["logvar_x"])
        loss_margin = self.margin_loss(out["z_x"], out["z_y"], self.margin_radius)
        total = loss_recon + loss_mat_ce + self.beta * loss_kl + self.gamma * loss_margin
        return {"loss": total, "loss_recon": loss_recon, "loss_mat_ce": loss_mat_ce,
                "loss_kl": loss_kl, "loss_margin": loss_margin}

    @torch.no_grad()
    def generate(
        self, target_curve: torch.Tensor, n_samples: int = 16, tau: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Encode target → Z_y, sample n_samples within B(Z_y, r), decode each."""
        self.eval()
        if target_curve.dim() == 1:
            target_curve = target_curve.unsqueeze(0)
        z_y = self.spectrum_encoder(target_curve)
        direction = F.normalize(torch.randn(n_samples, z_y.shape[-1], device=z_y.device), p=2, dim=-1)
        radii = torch.rand(n_samples, 1, device=z_y.device) * self.margin_radius
        z_samples = z_y.expand(n_samples, -1) + radii * direction
        pred_geo, mat_oh, mat_logits = self.geometry_decoder(z_samples, tau=tau, hard=True)
        return {"z_samples": z_samples, "pred_geometry": pred_geo,
                "material_onehot": mat_oh, "material_logits": mat_logits, "z_y": z_y}


class GratingDataset(torch.utils.data.Dataset):
    """Multi-material grating dataset from batched .pt files.

    Input inputs are dynamically scaled to [0, 1] using dataset statistics.
    For validation datasets, pass `geo_min` and `geo_max` from the train split.
    """

    def __init__(
        self, data_files: Dict[str, list[str]], target_key: str = "A_film_normal",
        geo_min: Optional[torch.Tensor] = None, geo_max: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        all_geometry: list[torch.Tensor] = []
        all_params_x: list[torch.Tensor] = []
        all_material: list[torch.Tensor] = []
        all_target: list[torch.Tensor] = []

        for mat_name, batch_files in data_files.items():
            mat_id = MATERIAL_LIBRARY[mat_name]
            if not batch_files:
                raise FileNotFoundError(f"No batch_*.pt files provided for {mat_name}")
            for bf in batch_files:
                data = torch.load(bf, map_location="cpu", weights_only=False)
                B = data["h"].shape[0]

                def process_target(key, override_inc_ang=None):
                    target = data[key].float()
                    if target.dim() == 2 and target.shape[1] == 2:
                        target = torch.cat([target[:, 0], target[:, 1]], dim=-1)
                    elif target.dim() == 3:
                        target = torch.cat([target[:, :, 0], target[:, :, 1]], dim=-1)

                    valid_mask = (target.max(dim=-1).values <= 1.0) & (target.min(dim=-1).values >= 0.0)
                    
                    if valid_mask.any():
                        px = data["params_x"].float()[valid_mask]
                        all_params_x.append(px)

                        geo_parts = [px.view(px.shape[0], -1)]
                        geo_parts.append(data["h"].float()[valid_mask].unsqueeze(-1))
                        if override_inc_ang is not None:
                            geo_parts.append(torch.full((valid_mask.sum().item(), 1), override_inc_ang, dtype=torch.float32))
                        else:
                            geo_parts.append(data["inc_ang"].float()[valid_mask].unsqueeze(-1))
                        all_geometry.append(torch.cat(geo_parts, dim=-1))

                        all_material.append(torch.full((valid_mask.sum().item(),), mat_id, dtype=torch.long))
                        all_target.append(target[valid_mask])

                if target_key == "all_film":
                    process_target("A_film_normal", override_inc_ang=0.0)
                    process_target("A_film_oblique", override_inc_ang=None)
                else:
                    process_target(target_key, override_inc_ang=None)
        
        self.geometry = torch.cat(all_geometry, dim=0)
        self.params_x = torch.cat(all_params_x, dim=0)
        self.material_id = torch.cat(all_material, dim=0)
        self.target = torch.cat(all_target, dim=0)
        
        # Record inputs dynamically based on actual dataset ranges (no normalization)
        if geo_min is None or geo_max is None:
            self.geo_min = self.geometry.min(dim=0).values
            self.geo_max = self.geometry.max(dim=0).values
        else:
            self.geo_min = geo_min
            self.geo_max = geo_max

        self._n_wavelengths = self.target.shape[-1]
        self._n_continuous = self.geometry.shape[-1]

    def __len__(self) -> int:
        return self.geometry.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"geometry": self.geometry[idx], "params_x": self.params_x[idx],
                "material_id": self.material_id[idx], "target": self.target[idx]}




class _EMATracker:
    """Lightweight EMA of model *parameters* (buffers like BN running stats are left untouched).

    After each optimizer step call ``update(model)``.  When validation improves call
    ``snapshot()`` to save the current EMA weights.  At the end of training call
    ``restore(model)`` to write the best EMA snapshot back into the model.

    Args:
        model:  The model whose parameters should be tracked.
        decay:  EMA decay factor (e.g. 0.999).  Higher = slower update, more smoothing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Initialise shadow from current model parameters
        self._shadow: dict[str, torch.Tensor] = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
        }
        self._best_snapshot: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Step the EMA after one optimiser update."""
        d = self.decay
        for name, param in model.named_parameters():
            self._shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def snapshot(self) -> None:
        """Save the current EMA shadow as the best seen so far."""
        self._best_snapshot = {k: v.clone() for k, v in self._shadow.items()}

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Write the best EMA snapshot back into *model* parameters.

        If no snapshot was ever taken (e.g. training lasted 0 epochs) this is a no-op.
        """
        src = self._best_snapshot if self._best_snapshot is not None else self._shadow
        for name, param in model.named_parameters():
            if name in src:
                param.data.copy_(src[name])


def train_forward_model(
    model: nn.Module, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, device: torch.device = torch.device("cpu"),
    use_bfloat16: bool = False,
    ema_decay: float = 0.999,
) -> dict[str, list[float]]:
    """Train Forward models (MLP, CNN, skipCNN, SIREN)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)
    criterion = nn.HuberLoss(delta=0.01)
    best_val, patience_ctr = float("inf"), 0
    ema = _EMATracker(model, decay=ema_decay)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        model.train()
        train_loss_accum = 0.0
        
        lambda_fft = 1.0
        
        for batch in train_loader:
            geo, mat, target = (batch["geometry"].to(device),
                                batch["material_id"].to(device), batch["target"].to(device))
                                
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16 and torch.cuda.is_available()):
                pred = model(geo, mat)
                
            # Compute loss in fp32 for stability and FFT compatibility
            pred_f32 = pred.float()
            target_f32 = target.float()
            
            base_loss = criterion(pred_f32, target_f32)
            wl = target_f32.shape[-1] // 2
            
            fft_pred_p = torch.fft.rfft(pred_f32[:, :wl], dim=-1, norm="ortho").abs()
            fft_pred_s = torch.fft.rfft(pred_f32[:, wl:], dim=-1, norm="ortho").abs()
            fft_target_p = torch.fft.rfft(target_f32[:, :wl], dim=-1, norm="ortho").abs()
            fft_target_s = torch.fft.rfft(target_f32[:, wl:], dim=-1, norm="ortho").abs()
            
            spectral_loss = criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)
            
            loss = base_loss + lambda_fft * spectral_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        model.eval()
        val_loss_accum = 0.0
        val_mae_accum = 0.0
        val_max_err_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                geo, mat, target = (batch["geometry"].to(device),
                                    batch["material_id"].to(device), batch["target"].to(device))
                pred = model(geo, mat)
                
                base_loss = criterion(pred, target)
                wl = target.shape[-1] // 2
                
                fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1, norm="ortho").abs()
                fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1, norm="ortho").abs()
                fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1, norm="ortho").abs()
                fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1, norm="ortho").abs()
                
                spectral_loss = criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)
                
                loss = base_loss + lambda_fft * spectral_loss
                
                val_loss_accum += loss.item()
                
                abs_err = torch.abs(pred - target)
                val_mae_accum += abs_err.mean().item()
                val_max_err_accum += abs_err.max(dim=1).values.mean().item()

        avg_val = val_loss_accum / len(val_loader)
        avg_mae = val_mae_accum / len(val_loader)
        avg_max_err = val_max_err_accum / len(val_loader)
        
        history["val_loss"].append(avg_val)
        history["val_mae"].append(avg_mae)
        history["val_max_err"].append(avg_max_err)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            ema.snapshot()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(f"lr={optimizer.param_groups[0]['lr']:.1e} "
                             f"best={best_val:.3e} train={avg_train:.3e} val={avg_val:.3e} "
                             f"vMAE={avg_mae:.3f} vMaxE={avg_max_err:.3f}")

    ema.restore(model)
    return history


def train_tandem(
    tandem: TandemNetwork | GenerativeTandemNetwork, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, tau_start: float = 1.0, tau_end: float = 0.1,
    device: torch.device = torch.device("cpu"),
    synthetic_phase: bool = False,
    ema_decay: float = 0.999,
) -> dict[str, list[float]]:
    """Train tandem. Only InverseDecoder params are optimised; forward model stays frozen."""
    tandem = tandem.to(device)
    optimizer = torch.optim.AdamW(tandem.inverse_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)
    criterion = nn.HuberLoss(delta=0.01)
    best_val, patience_ctr = float("inf"), 0
    ema = _EMATracker(tandem.inverse_decoder, decay=ema_decay)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(epochs - 1, 1)

        tandem.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            if synthetic_phase:
                target, weight_mask = generate_synthetic_targets(batch["target"].shape[0], batch["target"].shape[-1], device)
            else:
                target = batch["target"].to(device)
            out = tandem(target, tau=tau)
            pred = out["predicted_curve"]
            
            if synthetic_phase:
                loss = (torch.nn.functional.mse_loss(pred, target, reduction='none') * weight_mask).mean()
            else:
                base_loss = criterion(pred, target)
                wl = target.shape[-1] // 2
                # Spectral Loss (FFT Magnitude) - insensitive to small peak shifts
                fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1).abs()
                fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1).abs()
                fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1).abs()
                fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1).abs()
                
                spectral_loss = (criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)) / wl
                loss = base_loss + 0.5 * spectral_loss
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        tandem.eval()
        val_loss_accum = 0.0
        val_mae_accum = 0.0
        val_max_err_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if synthetic_phase:
                    target, weight_mask = generate_synthetic_targets(batch["target"].shape[0], batch["target"].shape[-1], device)
                else:
                    target = batch["target"].to(device)
                out = tandem(target, tau=tau)
                pred = out["predicted_curve"]
                
                if synthetic_phase:
                    loss = (torch.nn.functional.mse_loss(pred, target, reduction='none') * weight_mask).mean()
                    val_loss_accum += loss.item()
                else:
                    base_loss = criterion(pred, target)
                    wl = target.shape[-1] // 2
                    fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1).abs()
                    fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1).abs()
                    fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1).abs()
                    fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1).abs()
                    
                    spectral_loss = (criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)) / wl
                    val_loss_accum += (base_loss + 0.5 * spectral_loss).item()
                
                abs_err = torch.abs(pred - target)
                val_mae_accum += abs_err.mean().item()
                val_max_err_accum += abs_err.max(dim=1).values.mean().item()

        avg_val = val_loss_accum / len(val_loader)
        avg_mae = val_mae_accum / len(val_loader)
        avg_max_err = val_max_err_accum / len(val_loader)
        
        history["val_loss"].append(avg_val)
        history["val_mae"].append(avg_mae)
        history["val_max_err"].append(avg_max_err)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            ema.snapshot()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(f"tau={tau:.2f} lr={optimizer.param_groups[0]['lr']:.1e} "
                             f"best={best_val:.3e} train={avg_train:.3e} val={avg_val:.3e} "
                             f"vMAE={avg_mae:.3f} vMaxE={avg_max_err:.3f}")

    ema.restore(tandem.inverse_decoder)
    return history


def train_cvae(
    cvae: ContrastiveVAE, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, tau_start: float = 1.0, tau_end: float = 0.1,
    device: torch.device = torch.device("cpu"),
    forward_model: Optional[nn.Module] = None,
    synthetic_phase: bool = False,
    ema_decay: float = 0.999,
) -> dict[str, list[float]]:
    """Train C-VAE. All sub-networks trained jointly. Loss = recon + CE + beta·KL + gamma·margin."""
    cvae = cvae.to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)
    best_val, patience_ctr = float("inf"), 0
    ema = _EMATracker(cvae, decay=ema_decay)
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "train_mat_ce": [], "train_kl": [], "train_margin": [],
    }

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(epochs - 1, 1)

        cvae.train()
        accum = {"loss": 0.0, "recon": 0.0, "mat_ce": 0.0, "kl": 0.0, "margin": 0.0}
        for batch in train_loader:
            if synthetic_phase:
                assert forward_model is not None, "Forward model required for synthetic phase in CVAE"
                target, weight_mask = generate_synthetic_targets(batch["target"].shape[0], batch["target"].shape[-1], device)
                z_y = cvae.spectrum_encoder(target)
                pred_geo, mat_oh, _ = cvae.geometry_decoder(z_y, tau=tau, hard=True)
                pred_curve = forward_model(geometry=pred_geo, material_id=mat_oh)
                loss = (F.huber_loss(pred_curve, target, delta=0.01, reduction='none') * weight_mask).mean()
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                ema.update(cvae)
                accum["loss"] += loss.item()
            else:
                geo, mat, target = (batch["geometry"].to(device),
                                    batch["material_id"].to(device), batch["target"].to(device))
                out = cvae(geo, mat, target, tau=tau)
                losses = cvae.compute_loss(out, geo, mat)
                optimizer.zero_grad(); losses["loss"].backward(); optimizer.step()
                ema.update(cvae)
                for k in accum:
                    accum[k] += losses[f"loss_{k}"].item() if k != "loss" else losses["loss"].item()

        n = len(train_loader)
        history["train_loss"].append(accum["loss"] / n)
        for k in ("recon", "mat_ce", "kl", "margin"):
            history[f"train_{k}"].append(accum[k] / n)

        cvae.eval()
        val_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if synthetic_phase:
                    target, weight_mask = generate_synthetic_targets(batch["target"].shape[0], batch["target"].shape[-1], device)
                    z_y = cvae.spectrum_encoder(target)
                    pred_geo, mat_oh, _ = cvae.geometry_decoder(z_y, tau=tau, hard=True)
                    pred_curve = forward_model(geometry=pred_geo, material_id=mat_oh)
                    loss = (F.huber_loss(pred_curve, target, delta=0.01, reduction='none') * weight_mask).mean()
                    val_accum += loss.item()
                else:
                    geo, mat, target = (batch["geometry"].to(device),
                                        batch["material_id"].to(device), batch["target"].to(device))
                    losses = cvae.compute_loss(cvae(geo, mat, target, tau=tau), geo, mat)
                    val_accum += losses["loss"].item()

        avg_val = val_accum / len(val_loader)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            ema.snapshot()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix(total=f"{history['train_loss'][-1]:.3e}",
                         val=f"{avg_val:.3e}", kl=f"{accum['kl']/n:.3e}",
                         margin=f"{accum['margin']/n:.3e}", tau=f"{tau:.2f}")

    ema.restore(cvae)
    return history

def train_cvae_wishful(
    cvae: ContrastiveVAE, train_loader, val_loader, *,
    epochs: int = 100, lr: float = 1e-4, weight_decay: float = 1e-5,
    patience: int = 50, tau_start: float = 0.5, tau_end: float = 0.1,
    alpha_end: float = 0.99, top_k_quantile: float = 0.9,
    device: torch.device = torch.device("cpu"),
    ema_decay: float = 0.999,
) -> dict[str, list[float]]:
    """Wishful thinking finetuning: select top quantile of curves and gradually pull them to 1.0."""
    cvae = cvae.to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)
    best_val, patience_ctr = float("inf"), 0
    ema = _EMATracker(cvae, decay=ema_decay)
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "train_mat_ce": [], "train_kl": [], "train_margin": [],
    }

    pbar = tqdm(range(1, epochs + 1), desc="Finetune CVAE", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(epochs - 1, 1)
        alpha = alpha_end * (epoch - 1) / max(epochs - 1, 1)

        cvae.train()
        accum = {"loss": 0.0, "recon": 0.0, "mat_ce": 0.0, "kl": 0.0, "margin": 0.0}
        n_batches = 0
        for batch in train_loader:
            geo, mat, target = (batch["geometry"].to(device),
                                batch["material_id"].to(device), batch["target"].to(device))
            
            means = target.mean(dim=-1)
            threshold = torch.quantile(means, top_k_quantile)
            mask = means >= threshold
            if not mask.any(): continue
            
            geo, mat, target = geo[mask], mat[mask], target[mask]
            
            # Wishful thinking: shift curve towards 1.0
            target = target + alpha * (1.0 - target)
            
            out = cvae(geo, mat, target, tau=tau)
            losses = cvae.compute_loss(out, geo, mat)
            optimizer.zero_grad(); losses["loss"].backward(); optimizer.step()
            for k in accum:
                accum[k] += losses[f"loss_{k}"].item() if k != "loss" else losses["loss"].item()
            n_batches += 1

        if n_batches == 0: continue

        history["train_loss"].append(accum["loss"] / n_batches)
        for k in ("recon", "mat_ce", "kl", "margin"):
            history[f"train_{k}"].append(accum[k] / n_batches)

        cvae.eval()
        val_accum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                geo, mat, target = (batch["geometry"].to(device),
                                    batch["material_id"].to(device), batch["target"].to(device))
                means = target.mean(dim=-1)
                threshold = torch.quantile(means, top_k_quantile)
                mask = means >= threshold
                if not mask.any(): continue
                geo, mat, target = geo[mask], mat[mask], target[mask]
                target = target + alpha * (1.0 - target)
                
                losses = cvae.compute_loss(cvae(geo, mat, target, tau=tau), geo, mat)
                val_accum += losses["loss"].item()
                n_val += 1

        if n_val == 0: continue
        avg_val = val_accum / n_val
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            ema.snapshot()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix(total=f"{history['train_loss'][-1]:.3e}",
                         val=f"{avg_val:.3e}", alpha=f"{alpha:.2f}", tau=f"{tau:.2f}")

    ema.restore(cvae)
    return history



