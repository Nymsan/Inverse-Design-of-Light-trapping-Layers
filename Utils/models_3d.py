"""
SkipCNN3D — forward surrogate for 3D (bicontinuous) grating datasets.

Architecture
------------
The actual 3D grating surface is z(x, y) = profile_x(x) + profile_y(y),
an outer-sum on an (nx, ny) grid (matching get_continuous_boundary in utils.py).
Two independent 1-D CNNs cannot see cross-xy coupling, so we build the full
2-D height map and process it with a ResNet of 2-D circular-padded convolutions.

Input
-----
  params_x  : (B, 5, 2)  — [amplitude, phase] for x-harmonics
  params_y  : (B, 5, 2)  — [amplitude, phase] for y-harmonics
  h         : (B,)       — bulk Si thickness [nm]
  wavelength: (B,)       — queried wavelength [nm]

Output
------
  (B, 2) — [A_film_p, A_film_s]  (p- and s-pol absorptance in the bulk film)

Dataset keys (from generate_3d_dataset.py / consolidate_3d_dataset.py):
  params_x, params_y, h, wavelength,
  A_film_normal (B, 2), A_grating_normal (B, 2),
  A_film_max_wl (B, 2), A_grating_max_wl (B, 2)

Exports
-------
  build_2d_profile  — construct the (nx, ny) height map used by the simulator
  ResBlock2D        — 2-D residual conv block with circular padding
  SkipCNN3D         — forward surrogate model
  Grating3DDataset  — PyTorch Dataset for consolidated 3D batches
"""
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 2-D profile builder  (mirrors get_continuous_boundary in utils.py exactly)
# ---------------------------------------------------------------------------

def build_2d_profile(
    params_x: torch.Tensor,
    params_y: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    harmonic_idx: torch.Tensor,
    grating_period: float,
) -> torch.Tensor:
    """
    Build the 2-D grating surface z(x, y) = profile_x(x) + profile_y(y).

    Matches get_continuous_boundary() / _compute_1d_profile() in utils.py:
        profile_x(x) = H_x/2 + sum_n  A_x_n * cos(2π n x / Lx - φ_x_n)
        profile_y(y) = H_y/2 + sum_m  A_y_m * cos(2π m y / Ly - φ_y_m)
        z(x, y)      = profile_x(x) + profile_y(y)          (outer sum)

    Parameters
    ----------
    params_x, params_y : (B, H, 2)  — [:, :, 0]=amplitudes, [:, :, 1]=phases
    x_grid, y_grid     : (nx,), (ny,) — spatial sample points
    harmonic_idx       : (H,)  — [1, 2, …, H]
    grating_period     : scalar period (same for both axes)

    Returns
    -------
    surface : (B, 1, nx, ny)  — height map ready for 2-D convolution
    """
    B, H = params_x.shape[:2]
    nx, ny = x_grid.shape[0], y_grid.shape[0]

    def _profile_1d(params, r_grid):
        # grating_height = 2 * sum(amps) + eps   (matches _compute_1d_profile)
        amps   = params[:, :, 0]                           # (B, H)
        phases = params[:, :, 1]                           # (B, H)
        height = 2.0 * amps.sum(dim=1, keepdim=True) + 1e-9  # (B, 1)

        n = harmonic_idx[None, :, None]                    # (1, H, 1)
        r = r_grid[None, None, :]                          # (1, 1, L)
        arg = 2.0 * math.pi * n * r / grating_period - phases[:, :, None]
        cosines = amps[:, :, None] * torch.cos(arg)        # (B, H, L)
        return height / 2.0 + cosines.sum(dim=1)           # (B, L)

    prof_x = _profile_1d(params_x, x_grid)   # (B, nx)
    prof_y = _profile_1d(params_y, y_grid)   # (B, ny)

    # outer sum  →  (B, nx, ny)
    surface = prof_x.unsqueeze(2) + prof_y.unsqueeze(1)
    return surface.unsqueeze(1)               # (B, 1, nx, ny)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SkipLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm   = nn.LayerNorm(out_features)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.skip   = (nn.Linear(in_features, out_features, bias=False)
                       if in_features != out_features else nn.Identity())

    def forward(self, x):
        return self.drop(self.act(self.norm(self.linear(x)))) + self.skip(x)


class ResBlock2D(nn.Module):
    """
    2-D residual conv block with circular padding and optional 2× downsampling.

    Circular padding in both spatial dims matches the periodic boundary
    conditions of the grating unit cell.
    """

    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=0.0, downsample=False):
        super().__init__()
        stride    = 2 if downsample else 1
        pad       = kernel_size // 2
        # stride=2 conv doesn't support circular padding in PyTorch; use zeros
        # for the strided conv and circular for the subsequent one.
        pad_mode1 = "zeros" if downsample else "circular"

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=pad, padding_mode=pad_mode1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1,
                               padding=pad, padding_mode="circular")
        self.norm2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or downsample:
            layers = []
            if downsample:
                layers.append(nn.AvgPool2d(2, stride=2, ceil_mode=True))
            layers += [nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)]
            self.skip = nn.Sequential(*layers)
        else:
            self.skip = nn.Identity()

        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout2d(dropout)

    def forward(self, x):
        res = self.skip(x)
        x   = self.drop1(self.act1(self.norm1(self.conv1(x))))
        x   = self.norm2(self.conv2(x))
        return self.drop2(self.act2(x + res))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SkipCNN3D(nn.Module):
    """
    SkipCNN-style forward surrogate for 3D grating data.

    Builds the exact same 2-D height map z(x,y) = profile_x(x) + profile_y(y)
    that the RCWA simulator uses (outer sum, matching get_continuous_boundary),
    then encodes it with a stack of 2-D circular-padded residual conv blocks.
    The flattened feature map is combined with BatchNorm-normalised h and
    wavelength scalars before the MLP head predicts [A_p, A_s].

    Default parameters give ~1.0 M weights, matching the SkipCNN 2-D budget.
    """

    def __init__(
        self,
        n_harmonics:    int   = 5,
        nx:             int   = 64,        # 2-D grid is nx×nx; keep small to save memory
        grating_period: float = 1000.0,
        conv_channels:  tuple = (16, 32, 64, 32),
        kernel_size:    int   = 5,
        fc_dims:        tuple = (256, 128),
        dropout:        float = 0.0,
        n_outputs:      int   = 2,         # [p-pol, s-pol]
    ):
        super().__init__()
        self.n_harmonics    = n_harmonics
        self.nx             = nx
        self.grating_period = grating_period

        self.register_buffer("x_grid",       torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("y_grid",       torch.linspace(0, grating_period, nx + 1)[:-1])
        self.register_buffer("harmonic_idx", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

        # Scalar input normalisation
        self.h_norm  = nn.BatchNorm1d(1)
        self.wl_norm = nn.BatchNorm1d(1)

        # 2-D CNN backbone
        in_ch  = 1   # single-channel height map
        layers = []
        for i, out_ch in enumerate(conv_channels):
            layers.append(ResBlock2D(in_ch, out_ch, kernel_size, dropout,
                                     downsample=(i < len(conv_channels) - 1)))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Infer spatial size after pooling
        with torch.no_grad():
            dummy      = torch.zeros(1, 1, nx, nx)
            spatial_hw = self.cnn(dummy).shape[-1]   # square, so just one dim

        # conv_channels[-1] * spatial_hw² + h + wavelength
        fc_in = conv_channels[-1] * spatial_hw * spatial_hw + 2

        head = []
        for fc_dim in fc_dims:
            head.append(SkipLinear(fc_in, fc_dim, dropout=dropout))
            fc_in = fc_dim
        head += [nn.Linear(fc_in, n_outputs), nn.Sigmoid()]
        self.fc_head = nn.Sequential(*head)

    # ------------------------------------------------------------------
    def forward(self, params_x, params_y, h, wavelength):
        """
        params_x  : (B, 5, 2)
        params_y  : (B, 5, 2)
        h         : (B,)  or (B, 1)
        wavelength: (B,)  or (B, 1)
        """
        B = params_x.shape[0]

        # Build full 2-D height map  →  (B, 1, nx, nx)
        surface = build_2d_profile(
            params_x, params_y,
            self.x_grid, self.y_grid,
            self.harmonic_idx, self.grating_period,
        )

        # 2-D CNN  →  flatten
        feat = self.cnn(surface).view(B, -1)

        # Inject scalar physics
        h_n  = self.h_norm(h.view(B, 1))
        wl_n = self.wl_norm(wavelength.view(B, 1))

        x = torch.cat([feat, h_n, wl_n], dim=1)
        return self.fc_head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Grating3DDataset(Dataset):
    """
    Dataset for consolidated 3D LHS batches (train_dataset.pt / val_dataset.pt).

    Pass a list of .pt file paths (one per material / folder). Samples from
    all files are concatenated. h and wavelength statistics (min/max) are
    recorded from the first file list so that val sets can be normalised
    consistently with training.

    Parameters
    ----------
    files       : list of paths to train_dataset.pt or val_dataset.pt
    target_key  : "A_film_normal" or "A_grating_normal"  (shape B×2)
    h_min/max   : override normalisation bounds (use train set's values for val)
    wl_min/max  : override normalisation bounds (use train set's values for val)
    """

    REF_WL: float = 495.0   # fixed reference wavelength always computed by the simulator

    # map from target_key to the companion key computed at REF_WL
    _COMPANION: dict[str, str] = {
        "A_film_normal":    "A_film_max_wl",
        "A_grating_normal": "A_grating_max_wl",
    }

    def __init__(
        self,
        files: list[str],
        target_key: str = "A_film_normal",
        h_min:  float | None = None,
        h_max:  float | None = None,
        wl_min: float | None = None,
        wl_max: float | None = None,
    ):
        super().__init__()
        px_list, py_list, h_list, wl_list, tgt_list = [], [], [], [], []

        companion_key = self._COMPANION.get(target_key)

        for path in files:
            d = torch.load(path, map_location="cpu", weights_only=False)

            tgt   = d[target_key].float()           # (B, 2)
            valid = (tgt.max(dim=1).values <= 1.0) & (tgt.min(dim=1).values >= 0.0)

            px   = d["params_x"].float()[valid]
            py   = d["params_y"].float()[valid]
            h    = d["h"].float()[valid]
            wl   = d["wavelength"].float()[valid]
            tgt  = tgt[valid]

            # ── primary: sampled wavelength ───────────────────────────────
            px_list.append(px);   py_list.append(py)
            h_list.append(h);     wl_list.append(wl)
            tgt_list.append(tgt)

            # ── companion: fixed 495 nm reference (free from same RCWA call)
            if companion_key and companion_key in d:
                tgt_ref = d[companion_key].float()[valid]          # (B_valid, 2)
                ref_valid = (tgt_ref.max(dim=1).values <= 1.0) & (tgt_ref.min(dim=1).values >= 0.0)
                px_list.append(px[ref_valid]);  py_list.append(py[ref_valid])
                h_list.append(h[ref_valid])
                # wavelength for these rows is always REF_WL
                wl_list.append(torch.full((ref_valid.sum().item(),), self.REF_WL))
                tgt_list.append(tgt_ref[ref_valid])

        self.params_x   = torch.cat(px_list,  dim=0)   # (N, 5, 2)
        self.params_y   = torch.cat(py_list,  dim=0)   # (N, 5, 2)
        self.h          = torch.cat(h_list,   dim=0)   # (N,)
        self.wavelength = torch.cat(wl_list,  dim=0)   # (N,)
        self.target     = torch.cat(tgt_list, dim=0)   # (N, 2)

        # Scalar normalisation statistics (BatchNorm lives inside the model)
        self.h_min  = float(h_min  if h_min  is not None else self.h.min())
        self.h_max  = float(h_max  if h_max  is not None else self.h.max())
        self.wl_min = float(wl_min if wl_min is not None else self.wavelength.min())
        self.wl_max = float(wl_max if wl_max is not None else self.wavelength.max())

    def __len__(self) -> int:
        return self.params_x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "params_x":   self.params_x[idx],    # (5, 2)
            "params_y":   self.params_y[idx],    # (5, 2)
            "h":          self.h[idx],            # scalar
            "wavelength": self.wavelength[idx],  # scalar
            "target":     self.target[idx],       # (2,)  [p-pol, s-pol]
        }
