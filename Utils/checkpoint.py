"""
Checkpoint utilities for saving and loading model checkpoints.

Saves architecture configuration alongside weights so that models can be
re-instantiated without hardcoded dimensions.  All ``load_*`` functions are
**backwards-compatible** with the old checkpoint format (no ``model_class``
or ``model_config`` keys) by falling back to the original hardcoded defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from Utils.models import (
    N_MATERIALS,
    ForwardMLP,
    SpatialCNN,
    SkipCNN,
    SIREN,
    TransformerForward,
    InverseDecoder,
    TandemNetwork,
    GenerativeTandemNetwork,
    GeometryEncoder,
    GeometryDecoder,
    SpectrumEncoder,
    ContrastiveVAE,
)

# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

FORWARD_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "ForwardMLP": ForwardMLP,
    "SpatialCNN": SpatialCNN,
    "SkipCNN": SkipCNN,
    "SIREN": SIREN,
    "TransformerForward": TransformerForward,
}

INVERSE_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "TandemNetwork": TandemNetwork,
    "GenerativeTandemNetwork": GenerativeTandemNetwork,
    "ContrastiveVAE": ContrastiveVAE,
}

# Map from checkpoint filename stem → (class name, legacy kwargs builder)
_FORWARD_FILENAME_TO_CLASS: dict[str, str] = {
    "forward_mlp": "ForwardMLP",
    "spatial_cnn": "SpatialCNN",
    "skip_cnn": "SkipCNN",
    "siren": "SIREN",
    "transformer_forward": "TransformerForward",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip ``_orig_mod.`` prefix left by ``torch.compile``."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def _safe_load_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load *state_dict* into *model*, expanding material-related tensors if needed.

    Handles any key whose suffix matches:
      - ``material_embedding.weight``  — row-wise expansion (new rows initialised from current model)
      - ``material_head.weight``       — row-wise expansion
      - ``material_head.bias``         — row-wise expansion
    This covers both top-level and nested keys (e.g. ``forward_model.material_embedding.weight``).
    """
    MATERIAL_SUFFIXES = (
        "material_embedding.weight",
        "material_head.weight",
        "material_head.bias",
    )
    current_sd = model.state_dict()
    for key, ckpt_val in list(state_dict.items()):
        if not any(key.endswith(sfx) for sfx in MATERIAL_SUFFIXES):
            continue
        if key not in current_sd:
            continue
        cur_val = current_sd[key]
        if ckpt_val.shape[0] < cur_val.shape[0]:
            # Expand: keep current model's rows, overwrite the ones from the checkpoint
            new_val = cur_val.clone()
            new_val[: ckpt_val.shape[0]] = ckpt_val
            state_dict[key] = new_val
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_forward_checkpoint(
    model: nn.Module,
    history: dict,
    path: str,
    model_class_name: str,
    model_kwargs: dict,
    use_bfloat16: bool = False,
) -> None:
    """Save a forward model checkpoint with full architecture configuration."""
    # Ensure kwargs are JSON-safe (convert tuples → lists for torch.save)
    torch.save(
        {
            "model_class": model_class_name,
            "model_config": model_kwargs,
            "model_state_dict": model.state_dict(),
            "history": history,
            "use_bfloat16": use_bfloat16,
        },
        path,
    )


def save_inverse_checkpoint(
    model: nn.Module,
    history: dict,
    path: str,
    model_class_name: str,
    model_config: dict,
    forward_model_name: str,
    phases_trained: list[str],
    use_bfloat16: bool = False,
) -> None:
    """Save an inverse model checkpoint with full architecture configuration."""
    torch.save(
        {
            "model_class": model_class_name,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "history": history,
            "forward_model_name": forward_model_name,
            "phases_trained": phases_trained,
            "use_bfloat16": use_bfloat16,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Legacy fallback kwargs for old checkpoints
# ---------------------------------------------------------------------------

def _legacy_forward_kwargs(
    class_name: str,
    n_continuous: int,
    n_wavelengths: int,
    n_harmonics: int,
) -> dict:
    """Return the hardcoded kwargs used by the original training scripts."""
    base = dict(
        n_wavelengths=n_wavelengths,
        n_materials=N_MATERIALS,
        embed_dim=8,
        n_harmonics=n_harmonics,
        nx=128,
        n_continuous=n_continuous,
    )
    if class_name == "ForwardMLP":
        return base
    if class_name == "SpatialCNN":
        return {**base, "conv_channels": (32, 64, 64, 64), "fc_dims": (512, 128)}
    if class_name == "SkipCNN":
        return {**base, "conv_channels": (32, 64, 128, 64), "fc_dims": (256, 256)}
    if class_name == "SIREN":
        return {**base, "conv_channels": (32, 64, 64)}
    if class_name == "TransformerForward":
        return {
            **base,
            "d_model": 128,
            "nhead": 4,
            "dim_feedforward": 512,
            "num_layers": 3,
            "dropout": 0.0,
        }
    raise ValueError(f"Unknown forward class name: {class_name}")


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_forward_model(
    path: str | Path,
    *,
    n_continuous: int | None = None,
    n_wavelengths: int | None = None,
    n_harmonics: int | None = None,
) -> tuple[nn.Module, dict, str]:
    """Load a forward model checkpoint.

    New-format checkpoints carry ``model_class`` and ``model_config``.
    Old-format checkpoints fall back to hardcoded defaults using the filename
    stem and the provided *n_continuous* / *n_wavelengths* / *n_harmonics*.

    Returns:
        (model, history, model_class_name)
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # --- Determine class name and kwargs --------------------------------
    if "model_class" in ckpt and "model_config" in ckpt:
        # New-format checkpoint
        class_name: str = ckpt["model_class"]
        model_kwargs: dict = ckpt["model_config"]
    else:
        # Legacy checkpoint – derive class from filename
        stem = path.stem  # e.g. "skip_cnn"
        class_name = _FORWARD_FILENAME_TO_CLASS.get(stem)
        if class_name is None:
            raise ValueError(
                f"Cannot infer model class from legacy checkpoint filename '{path.name}'. "
                f"Expected one of: {list(_FORWARD_FILENAME_TO_CLASS.keys())}"
            )
        if n_continuous is None or n_wavelengths is None or n_harmonics is None:
            raise ValueError(
                "Legacy checkpoint detected but n_continuous, n_wavelengths, "
                "and n_harmonics are required for backwards-compatible loading."
            )
        model_kwargs = _legacy_forward_kwargs(
            class_name, n_continuous, n_wavelengths, n_harmonics
        )

    # --- Instantiate and load weights -----------------------------------
    model_cls = FORWARD_MODEL_REGISTRY[class_name]
    model = model_cls(**model_kwargs)

    sd = _clean_state_dict(ckpt["model_state_dict"])
    _safe_load_state_dict(model, sd)

    history = ckpt.get("history", {})
    return model, history, class_name


def load_inverse_model(
    path: str | Path,
    forward_model: nn.Module | None = None,
    dataset_stats: dict | None = None,
    *,
    n_continuous: int | None = None,
    n_wavelengths: int | None = None,
) -> tuple[nn.Module, dict, dict]:
    """Load an inverse model checkpoint.

    New-format checkpoints carry ``model_class`` and ``model_config``.
    Old-format checkpoints fall back to the original hardcoded defaults
    using the filename stem.

    Parameters:
        path: Path to checkpoint file.
        forward_model: Frozen forward surrogate (needed for Tandem / GenTandem).
        dataset_stats: Dict with ``geo_min`` / ``geo_max`` tensors.
        n_continuous: Number of geometry features (legacy fallback only).
        n_wavelengths: Number of wavelength channels (legacy fallback only).

    Returns:
        (model, history, metadata) where metadata = {"forward_model_name": ..., "phases_trained": ...}
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    geo_min = dataset_stats.get("geo_min") if dataset_stats else None
    geo_max = dataset_stats.get("geo_max") if dataset_stats else None

    if "model_class" in ckpt and "model_config" in ckpt:
        # ---- New-format checkpoint ------------------------------------
        class_name: str = ckpt["model_class"]
        cfg: dict = ckpt["model_config"]

        if class_name == "TandemNetwork":
            inv_kwargs = dict(cfg["inverse_decoder"])
            if geo_min is not None:
                inv_kwargs["geo_min"] = geo_min
            if geo_max is not None:
                inv_kwargs["geo_max"] = geo_max
            decoder = InverseDecoder(**inv_kwargs)
            model = TandemNetwork(inverse_decoder=decoder, forward_model=forward_model)

        elif class_name == "GenerativeTandemNetwork":
            inv_kwargs = dict(cfg["inverse_decoder"])
            if geo_min is not None:
                inv_kwargs["geo_min"] = geo_min
            if geo_max is not None:
                inv_kwargs["geo_max"] = geo_max
            decoder = InverseDecoder(**inv_kwargs)
            model = GenerativeTandemNetwork(
                inverse_decoder=decoder,
                forward_model=forward_model,
                latent_dim=cfg["latent_dim"],
            )

        elif class_name == "ContrastiveVAE":
            geo_enc = GeometryEncoder(**cfg["geometry_encoder"])

            dec_kwargs = dict(cfg["geometry_decoder"])
            if geo_min is not None:
                dec_kwargs["geo_min"] = geo_min
            if geo_max is not None:
                dec_kwargs["geo_max"] = geo_max
            geo_dec = GeometryDecoder(**dec_kwargs)

            spec_enc = SpectrumEncoder(**cfg["spectrum_encoder"])
            model = ContrastiveVAE(
                geometry_encoder=geo_enc,
                geometry_decoder=geo_dec,
                spectrum_encoder=spec_enc,
                margin_radius=cfg.get("margin_radius", 1.0),
                beta=cfg.get("beta", 1e-3),
                gamma=cfg.get("gamma", 1.0),
            )
        else:
            raise ValueError(f"Unknown inverse model class: {class_name}")

        metadata = {
            "forward_model_name": ckpt.get("forward_model_name", ""),
            "phases_trained": ckpt.get("phases_trained", []),
        }

    else:
        # ---- Legacy checkpoint ----------------------------------------
        stem = path.stem  # e.g. "tandem", "generative_tandem", "cvae"
        if n_continuous is None or n_wavelengths is None:
            raise ValueError(
                "Legacy inverse checkpoint detected but n_continuous and "
                "n_wavelengths are required for backwards-compatible loading."
            )

        if stem == "tandem":
            class_name = "TandemNetwork"
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths,
                n_geometry=n_continuous,
                n_materials=N_MATERIALS,
                latent_dim=0,
                geo_min=geo_min,
                geo_max=geo_max,
            )
            model = TandemNetwork(inverse_decoder=decoder, forward_model=forward_model)

        elif stem == "generative_tandem":
            class_name = "GenerativeTandemNetwork"
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths,
                n_geometry=n_continuous,
                n_materials=N_MATERIALS,
                latent_dim=32,
                geo_min=geo_min,
                geo_max=geo_max,
            )
            model = GenerativeTandemNetwork(
                inverse_decoder=decoder,
                forward_model=forward_model,
                latent_dim=32,
            )

        elif stem == "cvae":
            class_name = "ContrastiveVAE"
            geo_enc = GeometryEncoder(
                n_continuous=n_continuous,
                n_materials=N_MATERIALS,
                embed_dim=8,
                latent_dim=64,
                fc_dims=(256, 256),
            )
            geo_dec = GeometryDecoder(
                latent_dim=64,
                n_geometry=n_continuous,
                n_materials=N_MATERIALS,
                geo_min=geo_min,
                geo_max=geo_max,
                hidden_dims=(256, 256),
            )
            spec_enc = SpectrumEncoder(
                n_wavelengths=n_wavelengths,
                latent_dim=64,
                conv_channels=(32, 64, 128, 64),
                fc_dims=(256, 256),
            )
            model = ContrastiveVAE(
                geometry_encoder=geo_enc,
                geometry_decoder=geo_dec,
                spectrum_encoder=spec_enc,
                margin_radius=1.0,
                beta=1e-3,
                gamma=1.0,
            )
        else:
            raise ValueError(
                f"Cannot infer inverse model class from legacy checkpoint filename '{path.name}'."
            )

        metadata = {
            "forward_model_name": "",
            "phases_trained": [],
        }

    # --- Load weights ---------------------------------------------------
    sd = _clean_state_dict(ckpt["model_state_dict"])
    _safe_load_state_dict(model, sd)

    history = ckpt.get("history", {})
    return model, history, metadata


# ---------------------------------------------------------------------------
# Best forward model selection
# ---------------------------------------------------------------------------

_FORWARD_FILENAMES = [
    "forward_mlp.pt",
    "spatial_cnn.pt",
    "skip_cnn.pt",
    "siren.pt",
    "transformer_forward.pt",
]


def get_best_forward_model(
    ckpt_dir: str | Path,
    n_continuous: int | None = None,
    n_wavelengths: int | None = None,
    n_harmonics: int | None = None,
) -> tuple[nn.Module | None, str | None, float]:
    """Scan *ckpt_dir* for forward model checkpoints and return the best.

    Returns:
        (model, model_name, best_val_loss)  or  (None, None, inf) if none found.
    """
    ckpt_dir = Path(ckpt_dir)
    best_model: nn.Module | None = None
    best_name: str | None = None
    best_loss = float("inf")

    for fname in _FORWARD_FILENAMES:
        p = ckpt_dir / fname
        if not p.exists():
            continue

        model, history, class_name = load_forward_model(
            p,
            n_continuous=n_continuous,
            n_wavelengths=n_wavelengths,
            n_harmonics=n_harmonics,
        )

        if "val_loss" in history and len(history["val_loss"]) > 0:
            val_loss = min(history["val_loss"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = p.stem  # e.g. "skip_cnn"
                best_model = model

    return best_model, best_name, best_loss
