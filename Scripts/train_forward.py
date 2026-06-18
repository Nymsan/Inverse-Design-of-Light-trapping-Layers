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
import glob
import random
import matplotlib.pyplot as plt
from pathlib import Path

import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader, random_split

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS, ForwardMLP, SpatialCNN, SkipCNN, SIREN, TransformerForward,
    GratingDataset,
    train_forward_model,
)

def save_checkpoint(model, history: dict, path: str, use_bfloat16: bool = False):
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "use_bfloat16": use_bfloat16,
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
    p.add_argument("--skip", nargs="*", default=[], choices=["mlp", "cnn", "skipcnn", "siren", "transformer"])
    return p.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_dirs = {mat: d for mat, d in zip(args.materials, args.data_dir)}
    run_name = "_".join(args.materials)
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {list(data_dirs.values())}")
    t0 = time.time()
    
    train_files = {mat: [] for mat in args.materials}
    val_files = {mat: [] for mat in args.materials}
    
    for mat_name, data_dir in data_dirs.items():
        batch_files = sorted(glob.glob(f"{data_dir}/batch_*.pt"))
        if not batch_files:
            raise FileNotFoundError(f"No batch_*.pt files in {data_dir}")
        random.shuffle(batch_files)
        n_val = max(1, int(len(batch_files) * args.val_split))
        val_files[mat_name] = batch_files[:n_val]
        train_files[mat_name] = batch_files[n_val:]
        
    train_set = GratingDataset(train_files, target_key=args.target_key)
    val_set = GratingDataset(val_files, target_key=args.target_key, geo_min=train_set.geo_min, geo_max=train_set.geo_max)
    
    print(f"Datasets loaded: Train {len(train_set)} samples, Val {len(val_set)} samples in {time.time() - t0:.1f} s")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    n_continuous = train_set.geometry.shape[-1]
    n_harmonics = train_set.params_x.shape[1]
    n_wavelengths = train_set.target.shape[-1]

    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={args.materials}")

    # Save dataset stats
    geo_min = train_set.geo_min
    geo_max = train_set.geo_max
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
            
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_bfloat16=False
        )
        elapsed = time.time() - t0
        timings["forward_mlp"] = elapsed
        all_history["forward_mlp"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "forward_mlp.pt"), use_bfloat16=False)

    if "cnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SpatialCNN")
        print("=" * 60)
        model = SpatialCNN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, conv_channels=(32, 64, 64, 64), fc_dims=(512, 128)
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_bfloat16=False
        )
        elapsed = time.time() - t0
        timings["spatial_cnn"] = elapsed
        all_history["spatial_cnn"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "spatial_cnn.pt"), use_bfloat16=False)

    if "skipcnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SkipCNN")
        print("=" * 60)
        model = SkipCNN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, conv_channels=(32, 64, 128, 64), fc_dims=(256, 256)
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_bfloat16=False
        )
        elapsed = time.time() - t0
        timings["skip_cnn"] = elapsed
        all_history["skip_cnn"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "skip_cnn.pt"), use_bfloat16=False)

    if "siren" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SIREN")
        print("=" * 60)
        model = SIREN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            conv_channels=(32, 64, 64), kernel_size=7, dropout=0.0, siren_hidden=(256, 256, 256), latent_dim=64, omega_0=30.0
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_bfloat16=False
        )
        elapsed = time.time() - t0
        timings["siren"] = elapsed
        all_history["siren"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "siren.pt"), use_bfloat16=True)

    if "transformer" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: TransformerForward")
        print("=" * 60)
        model = TransformerForward(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            d_model=128, nhead=4, dim_feedforward=512, num_layers=3, dropout=0.0, grating_period=1000.0
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, use_bfloat16=True
        )
        elapsed = time.time() - t0
        timings["transformer"] = elapsed
        all_history["transformer"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_checkpoint(model, hist, str(ckpt_dir / "transformer_forward.pt"), use_bfloat16=True)

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
