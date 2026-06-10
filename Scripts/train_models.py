#!/usr/bin/env python
"""
Train all surrogate models on generated grating data.

Usage:
    python Scripts/train_models.py --data_dir Data/LHS_Dataset_Si
    python Scripts/train_models.py --data_dir Data/LHS_Dataset_Si Data/LHS_Dataset_TiO2 --materials Si TiO2

Outputs saved to Checkpoints/<run_name>/:
    forward_mlp.pt, spatial_cnn.pt, tandem.pt, generative_tandem.pt, cvae.pt
    history.json        — training curves for all models
    dataset_stats.pt    — geo_min/geo_max for inference-time denormalization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS,
    ForwardMLP, SpatialCNN,
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE,
    GratingDataset,
    train_forward_model, train_tandem, train_cvae,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train all surrogate models")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="Paths to LHS dataset directories (one per material)")
    p.add_argument("--materials", nargs="+", default=None,
                   help="Material names matching data_dirs order. "
                        "If omitted, inferred from directory names (e.g. LHS_Dataset_Si -> Si)")
    p.add_argument("--target_key", default="A_film_normal",
                   help="Which absorptance key to train on")
    p.add_argument("--run_name", default=None,
                   help="Name for checkpoint directory. Default: auto-generated from materials")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Fraction of data used for validation")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--latent_dim_gen", type=int, default=32,
                   help="Latent dim for generative tandem")
    p.add_argument("--latent_dim_cvae", type=int, default=64,
                   help="Latent dim for contrastive VAE")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Device (auto-detected if omitted)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["mlp", "cnn", "tandem", "gen_tandem", "cvae"],
                   help="Skip specific models")
    return p.parse_args()


def infer_material(data_dir: str) -> str:
    """Extract material name from directory like 'LHS_Dataset_Si' or 'LHS_Dataset_TiO2_Ag'."""
    name = Path(data_dir).name
    for prefix in ("LHS_Dataset_", "Dataset_"):
        if name.startswith(prefix):
            mat = name[len(prefix):].split("_")[0]
            if mat in MATERIAL_LIBRARY:
                return mat
    raise ValueError(f"Cannot infer material from '{data_dir}'. Use --materials explicitly.")


def save_checkpoint(model: nn.Module, history: dict, path: str, extra: dict = None):
    """Save model weights, training history, and optional metadata."""
    payload = {
        "model_state_dict": model.state_dict(),
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build material → directory mapping
    if args.materials:
        assert len(args.materials) == len(args.data_dirs), \
            f"Got {len(args.materials)} materials but {len(args.data_dirs)} data dirs"
        mat_dirs = {m: d for m, d in zip(args.materials, args.data_dirs)}
    else:
        mat_dirs = {infer_material(d): d for d in args.data_dirs}
    print(f"Materials: {mat_dirs}")

    # Output directory
    run_name = args.run_name or "_".join(sorted(mat_dirs.keys()))
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = GratingDataset(data_dirs=mat_dirs, target_key=args.target_key)
    n_val = int(len(full_dataset) * args.val_frac)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    n_continuous = full_dataset._n_continuous
    n_wavelengths = full_dataset._n_wavelengths
    n_harmonics = (n_continuous - 2) // 2  # subtract h and inc_ang, div by 2 for cos/sin
    print(f"  n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  n_harmonics={n_harmonics}")

    # Save dataset statistics for inference-time denormalization
    stats_path = ckpt_dir / "dataset_stats.pt"
    torch.save({
        "geo_min": full_dataset.geo_min,
        "geo_max": full_dataset.geo_max,
        "n_continuous": n_continuous,
        "n_wavelengths": n_wavelengths,
        "n_harmonics": n_harmonics,
        "materials": mat_dirs,
        "target_key": args.target_key,
    }, stats_path)
    print(f"  Dataset stats saved: {stats_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    all_history = {}
    timings = {}

    # ── 1. ForwardMLP ──
    if "mlp" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ForwardMLP")
        print("=" * 60)
        model = ForwardMLP(
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            hidden_dims=(512, 1024, 1024, 1024, 512), activation="snake",
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_cnn=False,
        )
        elapsed = time.time() - t0
        timings["forward_mlp"] = elapsed
        all_history["forward_mlp"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "forward_mlp.pt"))

    # ── 2. SpatialCNN ──
    if "cnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SpatialCNN")
        print("=" * 60)
        model = SpatialCNN(
            n_harmonics=n_harmonics, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            n_pixels=256, conv_channels=(64, 128, 256, 512, 256),
            fc_dims=(512, 1024, 512),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_cnn=True,
        )
        elapsed = time.time() - t0
        timings["spatial_cnn"] = elapsed
        all_history["spatial_cnn"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "spatial_cnn.pt"))

    # ── 3. TandemNetwork (requires trained forward model) ──
    if "tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: TandemNetwork")
        print("=" * 60)

        # Load best forward model as frozen surrogate
        fwd_ckpt = ckpt_dir / "forward_mlp.pt"
        if not fwd_ckpt.exists():
            print("  SKIPPED — forward_mlp.pt not found (train MLP first)")
        else:
            forward_model = ForwardMLP(
                n_continuous=n_continuous, n_wavelengths=n_wavelengths,
                n_materials=N_MATERIALS, embed_dim=8,
                hidden_dims=(512, 1024, 1024, 1024, 512), activation="snake",
            )
            forward_model.load_state_dict(
                torch.load(fwd_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
            )
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=0,
                hidden_dims=(512, 1024, 1024, 1024, 512),
            )
            tandem = TandemNetwork(inverse_decoder=decoder, forward_model=forward_model)
            n_params = sum(p.numel() for p in tandem.inverse_decoder.parameters())
            print(f"  Trainable parameters (decoder only): {n_params:,}")

            t0 = time.time()
            hist = train_tandem(
                tandem, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                device=device,
            )
            elapsed = time.time() - t0
            timings["tandem"] = elapsed
            all_history["tandem"] = hist
            print(f"  Time: {elapsed / 60:.1f} min")

            save_checkpoint(tandem, hist, str(ckpt_dir / "tandem.pt"))

    # ── 4. GenerativeTandemNetwork ──
    if "gen_tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: GenerativeTandemNetwork")
        print("=" * 60)

        fwd_ckpt = ckpt_dir / "forward_mlp.pt"
        if not fwd_ckpt.exists():
            print("  SKIPPED — forward_mlp.pt not found (train MLP first)")
        else:
            forward_model = ForwardMLP(
                n_continuous=n_continuous, n_wavelengths=n_wavelengths,
                n_materials=N_MATERIALS, embed_dim=8,
                hidden_dims=(512, 1024, 1024, 1024, 512), activation="snake",
            )
            forward_model.load_state_dict(
                torch.load(fwd_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
            )
            latent_dim = args.latent_dim_gen
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=latent_dim,
                hidden_dims=(512, 1024, 1024, 1024, 512),
            )
            gen_tandem = GenerativeTandemNetwork(
                inverse_decoder=decoder, forward_model=forward_model,
                latent_dim=latent_dim,
            )
            n_params = sum(p.numel() for p in gen_tandem.inverse_decoder.parameters())
            print(f"  Trainable parameters (decoder only): {n_params:,}")

            t0 = time.time()
            hist = train_tandem(
                gen_tandem, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                device=device,
            )
            elapsed = time.time() - t0
            timings["generative_tandem"] = elapsed
            all_history["generative_tandem"] = hist
            print(f"  Time: {elapsed / 60:.1f} min")

            save_checkpoint(gen_tandem, hist, str(ckpt_dir / "generative_tandem.pt"))

    # ── 5. ContrastiveVAE ──
    if "cvae" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ContrastiveVAE")
        print("=" * 60)
        latent_dim = args.latent_dim_cvae
        geo_enc = GeometryEncoder(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=latent_dim, hidden_dims=(512, 1024, 512),
        )
        geo_dec = GeometryDecoder(
            latent_dim=latent_dim, n_geometry=n_continuous,
            n_materials=N_MATERIALS, hidden_dims=(512, 1024, 512),
        )
        spec_enc = SpectrumEncoder(
            n_wavelengths=n_wavelengths, latent_dim=latent_dim,
            hidden_dims=(512, 1024, 1024, 512),
        )
        cvae = ContrastiveVAE(
            geometry_encoder=geo_enc, geometry_decoder=geo_dec,
            spectrum_encoder=spec_enc, margin_radius=1.0,
            beta=1e-3, gamma=1.0,
        )
        n_params = sum(p.numel() for p in cvae.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_cvae(
            cvae, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        timings["cvae"] = elapsed
        all_history["cvae"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(cvae, hist, str(ckpt_dir / "cvae.pt"))

    # Save combined history
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nTraining history: {history_path}")

    # Print timing summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    total = 0.0
    for name, t in timings.items():
        print(f"  {name:25s}  {t / 60:6.1f} min")
        total += t
    print(f"  {'TOTAL':25s}  {total / 60:6.1f} min")


if __name__ == "__main__":
    main()
