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
import torch.nn as nn
import torch.nn.functional as F

# Ag is reflector-only and NOT available as a grating material.
MATERIAL_LIBRARY: Dict[str, int] = {"Si": 0, "TiO2": 1, "Si3N4": 2}
N_MATERIALS: int = len(MATERIAL_LIBRARY)


class Snake(nn.Module):
    """Snake activation: x + (1/a) sin²(ax). Learnable per-channel frequency."""

    def __init__(self, in_features: int, a_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((in_features,), a_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2


def polar_to_cartesian(params: torch.Tensor) -> torch.Tensor:
    """(B, N, 2) [amp, phase] → (B, 2N) [a·cosφ, ..., a·sinφ, ...]."""
    amps = params[:, :, 0]
    phases = params[:, :, 1]
    return torch.cat([amps * torch.cos(phases),
                      amps * torch.sin(phases)], dim=1)


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
    import math
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
        import numpy as np
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class MaterialDispersion(nn.Module):
    """Encodes material ID and wavelength into a dynamic dispersion embedding."""
    def __init__(self, n_materials=3, embed_dim=8, hidden_dim=32, omega_0=10.0):
        super().__init__()
        self.mat_embed = nn.Embedding(n_materials, embed_dim)
        self.wl_net = SineLayer(1, hidden_dim, is_first=True, omega_0=omega_0)
        self.proj = nn.Sequential(
            SkipLinear(embed_dim + hidden_dim, hidden_dim, activation="gelu", norm="layer"),
            nn.Linear(hidden_dim, embed_dim)
        )

    def get_base_embedding(self, material_id: torch.Tensor) -> torch.Tensor:
        if material_id.dim() == 2:
            return torch.matmul(material_id.float(), self.mat_embed.weight)
        return self.mat_embed(material_id.long())

    def forward(self, material_id: torch.Tensor, wls: torch.Tensor) -> torch.Tensor:
        """
        material_id: (B,)
        wls: (W,) or (B, W) or (B, W, 1)
        Returns: (B, W, embed_dim)
        """
        B = material_id.shape[0]
        if wls.dim() == 1:
            wls = wls.view(1, -1, 1).expand(B, -1, -1)
        elif wls.dim() == 2:
            wls = wls.unsqueeze(-1)
            if wls.shape[0] == 1:
                wls = wls.expand(B, -1, -1)
        
        W = wls.shape[1]
        
        base_m = self.get_base_embedding(material_id) # (B, embed_dim)
        base_m = base_m.unsqueeze(1).expand(B, W, -1) # (B, W, embed_dim)
        
        w_embed = self.wl_net(wls) # (B, W, hidden_dim)
        
        x = torch.cat([base_m, w_embed], dim=-1)
        return self.proj(x) # (B, W, embed_dim)



class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.05):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2, padding_mode="circular")
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout1d(dropout)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2, padding_mode="circular")
        self.norm2 = nn.BatchNorm1d(out_ch)
        
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.skip = nn.Identity()
            
        self.act2 = nn.GELU()
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
    A purely dense 'Conditioned Implicit Neural Representation'.
    Flattens the structural profile and concatenates it with wavelength.
    """
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=322, n_materials=3, embed_dim=8, 
                 siren_hidden=(256, 512, 512, 256), omega_0=10.0, **kwargs):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.seq_len = n_wavelengths // 2
        self.dispersion = MaterialDispersion(n_materials, embed_dim, omega_0=omega_0)
        
        geo_mat_dim = nx + 1 + 1 + embed_dim
        self.input_norm = nn.BatchNorm1d(geo_mat_dim)
        
        self.geo_proj = nn.Linear(geo_mat_dim, siren_hidden[0])
        self.wl_siren = SineLayer(1, siren_hidden[0], is_first=True, omega_0=omega_0)
        self.disp_proj = nn.Linear(embed_dim, siren_hidden[0])
        
        siren_layers = []
        siren_in_dim = siren_hidden[0]
        for h_dim in siren_hidden[1:]:
            siren_layers.append(SineLayer(siren_in_dim, h_dim, is_first=False, omega_0=omega_0))
            siren_in_dim = h_dim
            
        self.siren_decoder = nn.Sequential(*siren_layers)
        self.head = nn.Linear(siren_in_dim, 2)
        
    def forward(self, geometry, material_id, wls=None):
        B = geometry.shape[0]
        
        profile, h, inc_ang = build_profile(geometry, self.n_harmonics, self.nx)
        mat_base = self.dispersion.get_base_embedding(material_id)
        
        geo_mat = torch.cat([profile, h, mat_base, inc_ang], dim=-1)
        geo_mat = self.input_norm(geo_mat)
        geo_embed = self.geo_proj(geo_mat)
        
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
            
        wls_embed = self.wl_siren(wls_expanded)
        
        disp_embed = self.dispersion(material_id, wls_expanded)
        disp_embed = self.disp_proj(disp_embed)
        
        geo_expanded = geo_embed.unsqueeze(1).expand(-1, W, -1)
        siren_in = wls_embed + geo_expanded + disp_embed
        
        out = self.head(self.siren_decoder(siren_in))
        
        if return_flat:
            return torch.cat([out[..., 0], out[..., 1]], dim=1)
        return out

class ForwardMLP(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, hidden_dims=(256, 512, 512, 256), activation="gelu", norm="layer", dropout=0.05, grating_period=1000.0):
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
            layers.append(SkipLinear(in_dim, h_dim, activation=activation, norm=norm, dropout=dropout))
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_wavelengths)

    def forward(self, geometry, material_id):
        profile, h, inc_ang = build_profile(geometry, self.n_harmonics, self.nx, self.grating_period, self.r_grid, self.harmonic_idx)
        mat_embed = _embed_material(material_id, self.material_embedding)
        
        x_list = [profile, h, mat_embed, inc_ang]
        x = torch.cat(x_list, dim=-1)
        x = self.input_norm(x)
        return self.head(self.trunk(x))


class SkipCNN(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, grating_period=1000.0, conv_channels=(32, 64, 64), kernel_size=7, fc_dims=(256, 128), dropout=0.05):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.grating_period = grating_period
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_ch = 1 + 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout))
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool1d(2)) # Downsample spatial dim
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        # After len(conv_channels)-1 MaxPool1d(2) ops, spatial dim is downsampled
        downsample_factor = 2 ** (len(conv_channels) - 1)
        spatial_dim = nx // downsample_factor
        fc_in = conv_channels[-1] * spatial_dim
        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, geometry, material_id):
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
        x = x.view(B, -1) # Flatten instead of mean()!
        return self.fc_head(x)

class SpatialCNN(nn.Module):
    def __init__(self, n_harmonics=5, nx=128, n_continuous=12, n_wavelengths=161, n_materials=3, embed_dim=8, grating_period=1000.0, conv_channels=(32, 64, 64), kernel_size=7, fc_dims=(256, 128), dropout=0.05):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.nx = nx
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.grating_period = grating_period
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        
        self.register_buffer("r_grid", torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        in_ch = 1 + 1 + embed_dim + 1
        self.input_norm = nn.BatchNorm1d(in_ch)
        
        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout1d(dropout),
            ]
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool1d(2)) # Downsample spatial dim
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        # After len(conv_channels)-1 MaxPool1d(2) ops, spatial dim is downsampled
        downsample_factor = 2 ** (len(conv_channels) - 1)
        spatial_dim = nx // downsample_factor
        fc_in = conv_channels[-1] * spatial_dim
        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, geometry, material_id):
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
        x = x.view(B, -1) # Flatten instead of mean()!
        return self.fc_head(x)

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
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout))
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)
        
        downsample_factor = 2 ** (len(conv_channels) - 1)
        spatial_dim = self.seq_len // downsample_factor
        fc_in = conv_channels[-1] * spatial_dim + latent_dim
        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(SkipLinear(fc_in, fc_dim, activation="gelu", norm="layer", dropout=dropout))
            fc_in = fc_dim
        self.fc_head = nn.Sequential(*fc_layers)

        self.geometry_head = nn.Linear(fc_in, n_geometry)
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
        
        pred_geometry_norm = self.geometry_head(h)
        pred_geometry = torch.sigmoid(pred_geometry_norm) * (self.geo_max - self.geo_min) + self.geo_min
        
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
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout))
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        downsample_factor = 2 ** (len(conv_channels) - 1)
        spatial_dim = nx // downsample_factor
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
        self.geometry_head = nn.Linear(in_dim, n_geometry)
        self.material_head = nn.Linear(in_dim, n_materials)

    def forward(self, z: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """Returns (recon_geometry_physical, material_onehot, material_logits)."""
        h = self.trunk(z)
        recon_geometry_norm = self.geometry_head(h)
        recon_geometry = torch.sigmoid(recon_geometry_norm) * (self.geo_max - self.geo_min) + self.geo_min
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
            conv_layers.append(ResBlock1D(in_ch, out_ch, kernel_size, dropout))
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)
        
        downsample_factor = 2 ** (len(conv_channels) - 1)
        spatial_dim = self.seq_len // downsample_factor
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
        
        loss_recon = F.mse_loss(pred_norm, targ_norm)
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
        self, data_dirs: Dict[str, str], target_key: str = "A_film_normal",
        geo_min: Optional[torch.Tensor] = None, geo_max: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        all_geometry: list[torch.Tensor] = []
        all_params_x: list[torch.Tensor] = []
        all_material: list[torch.Tensor] = []
        all_target: list[torch.Tensor] = []

        for mat_name, data_dir in data_dirs.items():
            mat_id = MATERIAL_LIBRARY[mat_name]
            import glob
            batch_files = sorted(glob.glob(f"{data_dir}/batch_*.pt"))
            if not batch_files:
                raise FileNotFoundError(f"No batch_*.pt files in {data_dir}")
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


class GratingWavelengthDataset(torch.utils.data.Dataset):
    """
    Wraps GratingDataset (or a Subset) to return exactly ONE wavelength point per sample.
    The returned target is [2] (P and S polarizations).
    The physical wavelength value (300 to 1100 nm) is provided as a separate key 'wavelength'.
    """
    def __init__(self, base_dataset: torch.utils.data.Dataset, n_wavelengths: int = 322):
        self.base = base_dataset
        self.n_wls = n_wavelengths // 2
        self.wl_vals = torch.linspace(300, 1100, self.n_wls, dtype=torch.float32) + 1e-3

    def __len__(self) -> int:
        return len(self.base) * self.n_wls

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = idx // self.n_wls
        wl_idx = idx % self.n_wls
        
        item = self.base[sample_idx]
        
        wl_val = self.wl_vals[wl_idx]
        
        target = item["target"]
        p_pol = target[wl_idx]
        s_pol = target[self.n_wls + wl_idx]
        target_wl = torch.stack([p_pol, s_pol])
        
        return {
            "geometry": item["geometry"],
            "wavelength": wl_val,
            "material_id": item["material_id"],
            "target": target_wl
        }


def train_forward_model(
    model: nn.Module, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Train Forward models (MLP, CNN, skipCNN, SIREN)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_ctr = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        model.train()
        train_loss_accum = 0.0
        
        lambda_fft = 1.0 * min(1.0, (epoch - 1) / max(1, epochs // 2))
        
        for batch in train_loader:
            geo, mat, target = (batch["geometry"].to(device),
                                batch["material_id"].to(device), batch["target"].to(device))
            pred = model(geo, mat)
            
            mse_loss = criterion(pred, target)
            wl = target.shape[-1] // 2
            
            fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1, norm="ortho").abs()
            fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1, norm="ortho").abs()
            fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1, norm="ortho").abs()
            fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1, norm="ortho").abs()
            
            spectral_loss = criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)
            
            loss = mse_loss + lambda_fft * spectral_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                
                mse_loss = criterion(pred, target)
                wl = target.shape[-1] // 2
                
                fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1, norm="ortho").abs()
                fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1, norm="ortho").abs()
                fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1, norm="ortho").abs()
                fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1, norm="ortho").abs()
                
                spectral_loss = criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)
                
                # Validation always evaluates against the final, full-weighted objective
                loss = mse_loss + 1.0 * spectral_loss
                
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(f"lr={optimizer.param_groups[0]['lr']:.1e} "
                             f"best={best_val:.3e} train={avg_train:.3e} val={avg_val:.3e} "
                             f"vMAE={avg_mae:.3f} vMaxE={val_max_err_accum:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_tandem(
    tandem: TandemNetwork | GenerativeTandemNetwork, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, tau_start: float = 1.0, tau_end: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Train tandem. Only InverseDecoder params are optimised; forward model stays frozen."""
    tandem = tandem.to(device)
    optimizer = torch.optim.AdamW(tandem.inverse_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_ctr = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(epochs - 1, 1)

        tandem.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            target = batch["target"].to(device)
            out = tandem(target, tau=tau)
            pred = out["predicted_curve"]
            
            mse_loss = criterion(pred, target)
            wl = target.shape[-1] // 2
            
            # Spectral Loss (FFT Magnitude) - insensitive to small peak shifts
            fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1).abs()
            fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1).abs()
            fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1).abs()
            fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1).abs()
            
            spectral_loss = (criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)) / wl
            
            loss = mse_loss + 0.5 * spectral_loss
            
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
                target = batch["target"].to(device)
                out = tandem(target, tau=tau)
                pred = out["predicted_curve"]
                
                mse_loss = criterion(pred, target)
                wl = target.shape[-1] // 2
                
                fft_pred_p = torch.fft.rfft(pred[:, :wl], dim=-1).abs()
                fft_pred_s = torch.fft.rfft(pred[:, wl:], dim=-1).abs()
                fft_target_p = torch.fft.rfft(target[:, :wl], dim=-1).abs()
                fft_target_s = torch.fft.rfft(target[:, wl:], dim=-1).abs()
                
                spectral_loss = (criterion(fft_pred_p, fft_target_p) + criterion(fft_pred_s, fft_target_s)) / wl
                
                val_loss_accum += (mse_loss + 0.5 * spectral_loss).item()
                
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
            best_state = {k: v.clone() for k, v in tandem.inverse_decoder.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(f"tau={tau:.2f} lr={optimizer.param_groups[0]['lr']:.1e} "
                             f"best={best_val:.3e} train={avg_train:.3e} val={avg_val:.3e} "
                             f"vMAE={avg_mae:.3f} vMaxE={val_max_err_accum:.3f}")

    if best_state is not None:
        tandem.inverse_decoder.load_state_dict(best_state)
    return history


def train_cvae(
    cvae: ContrastiveVAE, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, tau_start: float = 1.0, tau_end: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Train C-VAE. All sub-networks trained jointly. Loss = recon + CE + β·KL + γ·margin."""
    cvae = cvae.to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    best_val, best_state, patience_ctr = float("inf"), None, 0
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
            geo, mat, target = (batch["geometry"].to(device),
                                batch["material_id"].to(device), batch["target"].to(device))
            out = cvae(geo, mat, target, tau=tau)
            losses = cvae.compute_loss(out, geo, mat)
            optimizer.zero_grad(); losses["loss"].backward(); optimizer.step()
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
                geo, mat, target = (batch["geometry"].to(device),
                                    batch["material_id"].to(device), batch["target"].to(device))
                losses = cvae.compute_loss(cvae(geo, mat, target, tau=tau), geo, mat)
                val_accum += losses["loss"].item()

        avg_val = val_accum / len(val_loader)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in cvae.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix(total=f"{history['train_loss'][-1]:.3e}",
                         val=f"{avg_val:.3e}", kl=f"{accum['kl']/n:.3e}",
                         margin=f"{accum['margin']/n:.3e}", tau=f"{tau:.2f}")

    if best_state is not None:
        cvae.load_state_dict(best_state)
    return history

def train_cvae_wishful(
    cvae: ContrastiveVAE, train_loader, val_loader, *,
    epochs: int = 100, lr: float = 1e-4, weight_decay: float = 1e-5,
    patience: int = 50, tau_start: float = 0.5, tau_end: float = 0.1,
    alpha_end: float = 0.99, top_k_quantile: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Wishful thinking finetuning: select top quantile of curves and gradually pull them to 1.0."""
    cvae = cvae.to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    best_val, best_state, patience_ctr = float("inf"), None, 0
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
            best_state = {k: v.clone() for k, v in cvae.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix(total=f"{history['train_loss'][-1]:.3e}",
                         val=f"{avg_val:.3e}", alpha=f"{alpha:.2f}", tau=f"{tau:.2f}")

    if best_state is not None:
        cvae.load_state_dict(best_state)
    return history


def train_implicit_forward(
    model: SIREN, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Train SIREN implicitly directly on individual physical wavelengths."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_ctr = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    import sys
    from tqdm import tqdm
    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        model.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            geo = batch["geometry"].to(device)
            mat = batch["material_id"].to(device)
            wl = batch["wavelength"].to(device).unsqueeze(1) # shape [B, 1]
            target = batch["target"].to(device)
            
            # predict
            pred = model(geo, mat, wls=wl).squeeze(1)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        model.eval()
        val_loss_accum, val_mae_accum, val_max_err_accum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                geo = batch["geometry"].to(device)
                mat = batch["material_id"].to(device)
                wl = batch["wavelength"].to(device).unsqueeze(1)
                target = batch["target"].to(device)
                
                pred = model(geo, mat, wls=wl).squeeze(1)
                mse_loss = criterion(pred, target)
                val_loss_accum += mse_loss.item()
                
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(f"lr={optimizer.param_groups[0]['lr']:.1e} best={best_val:.3e} "
                             f"train={avg_train:.3e} val={avg_val:.3e} vMAE={avg_mae:.3f} vMaxE={avg_max_err:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
