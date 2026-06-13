#!/usr/bin/env python
"""
Train forward surrogate models on generated grating data.

Usage:
    python Scripts/train_forward.py --data_dir Data/LHS_Dataset_Si
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS, ForwardMLP, SpatialCNN, SkipCNN, SIREN,
    GratingDataset,
    train_forward_model,
)

def save_checkpoint(model, history: dict, path: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
    }, path)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", nargs="+", required=True)
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--target_key", type=str, default="all_film")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--skip", nargs="*", default=[], choices=["mlp", "cnn", "skipcnn", "siren"])
    return p.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dirs = {mat: d for mat, d in zip(args.materials, args.data_dir)}
    run_name = "_".join(args.materials)
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {list(data_dirs.values())}")
    t0 = time.time()
    dataset = GratingDataset(data_dirs, target_key=args.target_key)
    print(f"Dataset loaded: {len(dataset)} samples in {time.time() - t0:.1f} s")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    n_continuous = dataset.geometry.shape[-1]
    n_harmonics = dataset.params_x.shape[1]
    n_wavelengths = dataset.target.shape[-1]

    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={args.materials}")

    # Save dataset stats
    geo_min = dataset.geometry.min(dim=0).values
    geo_max = dataset.geometry.max(dim=0).values
    torch.save({
        "geo_min": geo_min,
        "geo_max": geo_max,
        "n_continuous": n_continuous,
        "n_wavelengths": n_wavelengths,
        "n_harmonics": n_harmonics,
        "materials": data_dirs,
        "target_key": args.target_key,
    }, ckpt_dir / "dataset_stats.pt")

    timings = {}
    all_history = {}

    if "mlp" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ForwardMLP")
        print("=" * 60)
        model = ForwardMLP(
            n_harmonics=n_harmonics, nx=128,
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            hidden_dims=(256, 512, 512, 256), activation="snake",
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        timings["forward_mlp"] = elapsed
        all_history["forward_mlp"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "forward_mlp.pt"))

    if "cnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SpatialCNN")
        print("=" * 60)
        model = SpatialCNN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, conv_channels=(32, 64, 128, 64),
            fc_dims=(256, 512, 256),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        timings["spatial_cnn"] = elapsed
        all_history["spatial_cnn"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "spatial_cnn.pt"))

    if "skipcnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SkipCNN")
        print("=" * 60)
        model = SkipCNN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, conv_channels=(32, 64, 128, 64), fc_dims=(256, 512, 256),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        timings["skip_cnn"] = elapsed
        all_history["skip_cnn"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "skip_cnn.pt"))

    if "siren" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SIREN")
        print("=" * 60)
        model = SIREN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            conv_channels=(32, 64, 128, 64), kernel_size=7, dropout=0.05, siren_hidden=(256, 256, 256), latent_dim=128, omega_0=10.0
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        timings["siren"] = elapsed
        all_history["siren"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "siren.pt"))

    history_path = ckpt_dir / "forward_history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nTraining history: {history_path}")

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
