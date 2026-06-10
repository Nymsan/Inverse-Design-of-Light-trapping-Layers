"""
Surrogate models for multi-material light-trapping inverse design.

Models:
    1. ForwardMLP              – MLP forward model with material embedding
    2. SpatialCNN              – 1D CNN constructing grating profile in forward pass
    3. TandemNetwork           – Deterministic inverse model
    4. GenerativeTandemNetwork – Conditional generative inverse model
    5. ContrastiveVAE          – VAE with margin loss for shared latent neighbourhood

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


class ForwardMLP(nn.Module):
    """MLP: geometry + embedded material → absorptance curve ∈ [0,1]."""

    def __init__(
        self,
        n_continuous: int = 12,
        n_wavelengths: int = 161,
        n_materials: int = N_MATERIALS,
        embed_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 512, 512, 256),
        activation: Literal["gelu", "relu", "snake"] = "snake",
        norm: Literal["batch", "layer"] = "layer",
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_wavelengths = n_wavelengths
        self.material_embedding = nn.Embedding(n_materials, embed_dim)

        in_dim = n_continuous + embed_dim
        layers: list[nn.Module] = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim) if norm == "batch" else nn.LayerNorm(h_dim))
            layers.append(_make_activation(activation, h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_wavelengths)

    def forward(self, geometry: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """geometry: (B, n_continuous), material_id: (B,) int or (B, N_mat) one-hot → (B, N_wl)."""
        mat_embed = _embed_material(material_id, self.material_embedding)
        x = torch.cat([geometry, mat_embed], dim=-1)
        return torch.sigmoid(self.head(self.trunk(x)))
class SpatialMamba(nn.Module):
    """Mamba forward model that scans the 1D grating profile sequentially."""

    def __init__(
        self,
        n_harmonics: int = 5,
        n_wavelengths: int = 161,
        n_materials: int = 3,
        embed_dim: int = 8,
        n_pixels: int = 256,
        grating_period: float = 1000.0,
        d_model: int = 512,
        n_layers: int = 4,
        fc_dims: Sequence[int] = (512, 1024, 512),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_pixels = n_pixels
        self.grating_period = grating_period

        self.register_buffer("r_grid", torch.linspace(0, grating_period, n_pixels + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.material_embedding = nn.Embedding(n_materials, embed_dim)

        from Utils.mamba import MambaNet
        # Input features per step: 1 (profile height) + embed_dim (material) + 1 (h)
        in_ch = 1 + embed_dim + 1
        self.mamba = MambaNet(d_input=in_ch, d_model=d_model, n_layers=n_layers)

        fc_in = d_model
        fc_layers: list[nn.Module] = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(fc_in, fc_dim))
            fc_layers.append(nn.LayerNorm(fc_dim))
            fc_layers.append(nn.GELU())
            fc_layers.append(nn.Dropout(dropout))
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        self.fc_head = nn.Sequential(*fc_layers)

    def _build_profile(self, params_x: torch.Tensor) -> torch.Tensor:
        amps = params_x[:, :, 0]
        phases = params_x[:, :, 1]
        grating_height = 2.0 * amps.sum(dim=1, keepdim=True) + 1e-9

        n = self.harmonic_idx[None, :, None]
        r = self.r_grid[None, None, :]
        arg = 2.0 * math.pi * n * r / self.grating_period - phases[:, :, None]
        cosines = amps[:, :, None] * torch.cos(arg)
        profile = grating_height[:, :, None] / 2.0 + cosines.sum(dim=1, keepdim=True)

        p_min = profile.min(dim=-1, keepdim=True).values
        p_max = profile.max(dim=-1, keepdim=True).values
        return (profile - p_min) / (p_max - p_min + 1e-9)

    def forward(self, params_x: torch.Tensor, h_norm: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        P = self.n_pixels
        profile = self._build_profile(params_x) # (B, 1, P)

        mat_embed = _embed_material(material_id, self.material_embedding) # (B, embed_dim)
        mat_channel = mat_embed.unsqueeze(-1).expand(-1, -1, P)

        if h_norm.dim() == 1:
            h_norm = h_norm.unsqueeze(-1)
        h_channel = h_norm.unsqueeze(-1).expand(-1, -1, P)

        x = torch.cat([profile, mat_channel, h_channel], dim=1) # (B, in_ch, P)
        
        # Mamba expects sequence: (B, P, in_ch)
        x_seq = x.transpose(1, 2)
        
        mamba_out = self.mamba(x_seq) # (B, P, d_model)
        h = mamba_out.mean(dim=1) # Global average pooling over spatial sequence
        
        return torch.sigmoid(self.fc_head(h))

class SpatialCNN(nn.Module):
    """1D CNN that builds a grating profile from Fourier params, then convolves.

    Mirrors Utils/utils.py::_compute_1d_profile but is batch-vectorised.
    """

    def __init__(
        self,
        n_harmonics: int = 5,
        n_wavelengths: int = 161,
        n_materials: int = N_MATERIALS,
        embed_dim: int = 8,
        n_pixels: int = 128,
        grating_period: float = 1000.0,
        conv_channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 7,
        fc_dims: Sequence[int] = (256, 128),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_pixels = n_pixels
        self.grating_period = grating_period

        self.register_buffer("r_grid", torch.linspace(0, grating_period, n_pixels + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.material_embedding = nn.Embedding(n_materials, embed_dim)

        # Channels: 1 (profile) + embed_dim (material) + 1 (h)
        in_ch = 1 + embed_dim + 1
        conv_layers: list[nn.Module] = []
        for out_ch in conv_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        fc_in = conv_channels[-1]
        fc_layers: list[nn.Module] = []
        for fc_dim in fc_dims:
            fc_layers += [nn.Linear(fc_in, fc_dim), nn.LayerNorm(fc_dim), nn.GELU(), nn.Dropout(dropout)]
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, n_wavelengths))
        self.fc_head = nn.Sequential(*fc_layers)

    def _build_profile(self, params_x: torch.Tensor) -> torch.Tensor:
        """Fourier-synthesise a 1D grating profile. (B, N, 2) → (B, 1, n_pixels) normalised to [0,1]."""
        amps = params_x[:, :, 0]
        phases = params_x[:, :, 1]
        grating_height = 2.0 * amps.sum(dim=1, keepdim=True) + 1e-9

        n = self.harmonic_idx[None, :, None]
        r = self.r_grid[None, None, :]
        arg = 2.0 * math.pi * n * r / self.grating_period - phases[:, :, None]
        cosines = amps[:, :, None] * torch.cos(arg)
        profile = grating_height[:, :, None] / 2.0 + cosines.sum(dim=1, keepdim=True)

        p_min = profile.min(dim=-1, keepdim=True).values
        p_max = profile.max(dim=-1, keepdim=True).values
        return (profile - p_min) / (p_max - p_min + 1e-9)

    def forward(self, params_x: torch.Tensor, h_norm: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        """params_x: (B,N,2), h_norm: (B,1), material_id: (B,) → (B, N_wl) in [0,1]."""
        P = self.n_pixels
        profile = self._build_profile(params_x)

        mat_embed = _embed_material(material_id, self.material_embedding)
        mat_channel = mat_embed.unsqueeze(-1).expand(-1, -1, P)

        if h_norm.dim() == 1:
            h_norm = h_norm.unsqueeze(-1)
        h_channel = h_norm.unsqueeze(-1).expand(-1, -1, P)

        x = torch.cat([profile, mat_channel, h_channel], dim=1)
        x = self.conv_backbone(x)
        x = x.mean(dim=-1)
        return torch.sigmoid(self.fc_head(x))


class InverseDecoder(nn.Module):
    """Maps absorptance curve (+ optional noise z) → normalized geometry [0,1] + Gumbel material."""

    def __init__(
        self,
        n_wavelengths: int = 161,
        n_geometry: int = 12,
        n_materials: int = 3,
        latent_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256, 256),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_geometry = n_geometry
        self.n_materials = n_materials
        self.latent_dim = latent_dim
        self.seq_len = n_wavelengths // 2

        from Utils.mamba import MambaNet
        # Input features per step: 2 (p-pol and s-pol) + latent_dim
        self.mamba = MambaNet(d_input=2 + latent_dim, d_model=hidden_dims[0], n_layers=4)

        in_dim = hidden_dims[0]
        layers: list[nn.Module] = []
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.trunk = nn.Sequential(*layers)

        self.geometry_head = nn.Linear(in_dim, n_geometry)
        self.material_head = nn.Linear(in_dim, n_materials)

    def forward(
        self, target_curve: torch.Tensor, z: Optional[torch.Tensor] = None,
        tau: float = 1.0, hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred_geometry in [0,1], material_onehot, material_logits)."""
        B = target_curve.shape[0]
        
        # Reshape target_curve from (B, N_wl) to (B, seq_len, 2)
        x_seq = target_curve.view(B, 2, self.seq_len).transpose(1, 2)
        
        if z is not None:
            # Expand z to (B, seq_len, latent_dim)
            z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1)
            x_seq = torch.cat([x_seq, z_seq], dim=-1)
            
        # Process with Mamba
        mamba_out = self.mamba(x_seq) # (B, seq_len, d_model)
        
        # Global average pooling
        h = mamba_out.mean(dim=1)
        
        # MLP Trunk
        h = self.trunk(h)
        
        pred_geometry = torch.sigmoid(self.geometry_head(h))
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
        self, n_continuous: int = 11, n_materials: int = N_MATERIALS, embed_dim: int = 8,
        latent_dim: int = 64, hidden_dims: Sequence[int] = (256, 256), dropout: float = 0.05,
    ):
        super().__init__()
        self.material_embedding = nn.Embedding(n_materials, embed_dim)
        in_dim = n_continuous + embed_dim
        layers: list[nn.Module] = []
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, geometry: torch.Tensor, material_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mat_embed = _embed_material(material_id, self.material_embedding)
        h = self.trunk(torch.cat([geometry, mat_embed], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class GeometryDecoder(nn.Module):
    """VAE decoder: Z_x → normalized geometry [0,1] + Gumbel material."""

    def __init__(
        self, latent_dim: int = 64, n_geometry: int = 11, n_materials: int = N_MATERIALS,
        hidden_dims: Sequence[int] = (256, 256), dropout: float = 0.05,
    ):
        super().__init__()
        self.n_geometry = n_geometry
        self.n_materials = n_materials

        in_dim = latent_dim
        layers: list[nn.Module] = []
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.geometry_head = nn.Linear(in_dim, n_geometry)
        self.material_head = nn.Linear(in_dim, n_materials)

    def forward(self, z: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """Returns (recon_geometry in [0,1], material_onehot, material_logits)."""
        h = self.trunk(z)
        recon_geometry = torch.sigmoid(self.geometry_head(h))
        material_logits = self.material_head(h)
        material_onehot = F.gumbel_softmax(material_logits, tau=tau, hard=hard)
        return recon_geometry, material_onehot, material_logits


class SpectrumEncoder(nn.Module):
    """Deterministic encoder: target curve → latent center Z_y."""

    def __init__(
        self, n_wavelengths: int = 161, latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 256), dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = n_wavelengths // 2
        
        from Utils.mamba import MambaNet
        # Input features per step: 2 (p-pol and s-pol)
        self.mamba = MambaNet(d_input=2, d_model=hidden_dims[0], n_layers=4)
        
        in_dim = hidden_dims[0]
        layers: list[nn.Module] = []
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, latent_dim)

    def forward(self, target_curve: torch.Tensor) -> torch.Tensor:
        B = target_curve.shape[0]
        # Reshape to (B, seq_len, 2)
        x_seq = target_curve.view(B, 2, self.seq_len).transpose(1, 2)
        
        mamba_out = self.mamba(x_seq) # (B, seq_len, d_model)
        h = mamba_out.mean(dim=1) # Global average pooling
        
        h = self.trunk(h)
        return self.head(h)


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
        """Z_x = μ + σ⊙ε, ε ~ N(0,I)."""
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
        loss_recon = F.mse_loss(out["recon_geometry"], geometry)
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

                target = data[target_key].float()
                if target.dim() == 2 and target.shape[1] == 2:
                    target = torch.cat([target[:, 0], target[:, 1]], dim=-1)
                elif target.dim() == 3:
                    target = torch.cat([target[:, :, 0], target[:, :, 1]], dim=-1)

                # Filter out exploding curves (unphysical RCWA artifacts)
                valid_mask = (target.max(dim=-1).values <= 1.0) & (target.min(dim=-1).values >= 0.0)
                
                if valid_mask.any():
                    px = data["params_x"].float()[valid_mask]
                    all_params_x.append(px)

                    geo_parts = [polar_to_cartesian(px)]
                    geo_parts.append(data["h"].float()[valid_mask].unsqueeze(-1))
                    if "inc_ang" in data:
                        geo_parts.append(data["inc_ang"].float()[valid_mask].unsqueeze(-1))
                    all_geometry.append(torch.cat(geo_parts, dim=-1))

                    all_material.append(torch.full((valid_mask.sum().item(),), mat_id, dtype=torch.long))
                    all_target.append(target[valid_mask])
        self.geometry = torch.cat(all_geometry, dim=0)
        self.params_x = torch.cat(all_params_x, dim=0)
        self.material_id = torch.cat(all_material, dim=0)
        self.target = torch.cat(all_target, dim=0)
        
        # Scale inputs dynamically based on actual dataset ranges
        if geo_min is None or geo_max is None:
            self.geo_min = self.geometry.min(dim=0).values
            self.geo_max = self.geometry.max(dim=0).values
        else:
            self.geo_min = geo_min
            self.geo_max = geo_max
            
        span = self.geo_max - self.geo_min
        span[span == 0] = 1.0  # prevent div by zero for constants
        self.geometry = (self.geometry - self.geo_min) / span

        self._n_wavelengths = self.target.shape[-1]
        self._n_continuous = self.geometry.shape[-1]

    def __len__(self) -> int:
        return self.geometry.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"geometry": self.geometry[idx], "params_x": self.params_x[idx],
                "material_id": self.material_id[idx], "target": self.target[idx]}


def train_forward_model(
    model: nn.Module, train_loader, val_loader, *,
    epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 100, device: torch.device = torch.device("cpu"), use_cnn: bool = False,
) -> dict[str, list[float]]:
    """Train ForwardMLP or SpatialCNN. use_cnn=True passes params_x/h separately."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 5)
    criterion = nn.MSELoss()
    best_val, best_state, patience_ctr = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        model.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            geo, px, mat, target = (batch["geometry"].to(device), batch["params_x"].to(device),
                                    batch["material_id"].to(device), batch["target"].to(device))
            h_val = geo[:, -1:]
            pred = model(px, h_val, mat) if use_cnn else model(geo, mat)
            loss = criterion(pred, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                geo, px, mat, target = (batch["geometry"].to(device), batch["params_x"].to(device),
                                        batch["material_id"].to(device), batch["target"].to(device))
                h_val = geo[:, -1:]
                pred = model(px, h_val, mat) if use_cnn else model(geo, mat)
                val_loss_accum += criterion(pred, target).item()

        avg_val = val_loss_accum / len(val_loader)
        history["val_loss"].append(avg_val)
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

        pbar.set_postfix(train=f"{avg_train:.3e}", val=f"{avg_val:.3e}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}", best=f"{best_val:.3e}")

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
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(epochs - 1, 1)

        tandem.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            target = batch["target"].to(device)
            out = tandem(target, tau=tau)
            loss = criterion(out["predicted_curve"], target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_accum += loss.item()

        avg_train = train_loss_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        tandem.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(device)
                out = tandem(target, tau=tau)
                val_loss_accum += criterion(out["predicted_curve"], target).item()

        avg_val = val_loss_accum / len(val_loader)
        history["val_loss"].append(avg_val)
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

        pbar.set_postfix(train=f"{avg_train:.3e}", val=f"{avg_val:.3e}",
                         tau=f"{tau:.2f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                         best=f"{best_val:.3e}")

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
