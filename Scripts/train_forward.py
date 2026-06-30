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
from Utils.checkpoint import save_forward_checkpoint

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../Data")
    p.add_argument("--dataset_prefixes", nargs="+", default=["LHS_Dataset"])
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--train_subset_fraction", type=float, default=1.0)
    p.add_argument("--target_key", type=str, default="all_film")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--skip", nargs="*", default=[], choices=["mlp", "cnn", "skipcnn", "siren", "transformer"])
    p.add_argument("--embed_dim", type=int, default=8)
    p.add_argument("--smooth_target", type=int, default=None, help="Window size for moving average smoothing")
    
    # Architecture arguments
    # MLP
    p.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[512, 768, 512])
    p.add_argument("--mlp_dropout", type=float, default=0.0)
    
    # SpatialCNN
    p.add_argument("--cnn_conv_channels", type=int, nargs="+", default=[64, 128, 128, 64])
    p.add_argument("--cnn_kernel_size", type=int, default=7)
    p.add_argument("--cnn_fc_dims", type=int, nargs="+", default=[512, 128])
    p.add_argument("--cnn_dropout", type=float, default=0.0)
    
    # SkipCNN
    p.add_argument("--skipcnn_conv_channels", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--skipcnn_kernel_size", type=int, default=7)
    p.add_argument("--skipcnn_fc_dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--skipcnn_dropout", type=float, default=0.0)
    
    # SIREN
    p.add_argument("--siren_conv_channels", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--siren_kernel_size", type=int, default=7)
    p.add_argument("--siren_fc_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--siren_latent_dim", type=int, default=64)
    p.add_argument("--siren_omega_0", type=float, default=30.0)
    p.add_argument("--siren_dropout", type=float, default=0.0)
    
    # Transformer
    p.add_argument("--tf_d_model", type=int, default=128)
    p.add_argument("--tf_nhead", type=int, default=4)
    p.add_argument("--tf_dim_feedforward", type=int, default=512)
    p.add_argument("--tf_num_layers", type=int, default=3)
    p.add_argument("--tf_dropout", type=float, default=0.0)
    
    return p.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = "_".join(args.materials)
    if args.train_subset_fraction < 1.0:
        run_name += f"_frac_{args.train_subset_fraction}"
    if args.smooth_target is not None and args.smooth_target > 1:
        run_name += f"_smoothed_{args.smooth_target}"
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data with prefixes: {args.dataset_prefixes}")
    t0 = time.time()
    
    train_files = {mat: [] for mat in args.materials}
    val_files = {mat: [] for mat in args.materials}
    
    for mat_name in args.materials:
        for prefix in args.dataset_prefixes:
            d_dir = os.path.join(args.data_dir, f"{prefix}_{mat_name}")
            t_file = os.path.join(d_dir, "train_dataset.pt")
            v_file = os.path.join(d_dir, "val_dataset.pt")
            if os.path.exists(t_file) and os.path.exists(v_file):
                train_files[mat_name].append(t_file)
                val_files[mat_name].append(v_file)
            else:
                raise FileNotFoundError(f"Missing train/val datasets in {d_dir}")
                
    train_set = GratingDataset(train_files, target_key=args.target_key)
    val_set = GratingDataset(val_files, target_key=args.target_key, geo_min=train_set.geo_min, geo_max=train_set.geo_max)
    
    # Apply train_subset_fraction
    if args.train_subset_fraction < 1.0:
        print(f"Applying train_subset_fraction {args.train_subset_fraction}")
        rng = torch.Generator().manual_seed(args.seed)
        n_train_total = len(train_set)
        indices = torch.randperm(n_train_total, generator=rng)
        n_train = max(1, int(n_train_total * args.train_subset_fraction))
        t_idx = indices[:n_train]
        
        train_set.geometry = train_set.geometry[t_idx]
        train_set.params_x = train_set.params_x[t_idx]
        train_set.material_id = train_set.material_id[t_idx]
        train_set.target = train_set.target[t_idx]
        print(f"Reduced training set to {len(train_set)} samples")
        
    if args.smooth_target is not None and args.smooth_target > 1:
        print(f"Applying moving average smoothing to targets with window {args.smooth_target}")
        window = args.smooth_target
        
        def smooth_targets(target_tensor):
            n_wl_half = target_tensor.shape[-1] // 2
            p_pol = target_tensor[..., :n_wl_half].unsqueeze(1)
            s_pol = target_tensor[..., n_wl_half:].unsqueeze(1)
            
            kernel = torch.ones(1, 1, window, dtype=target_tensor.dtype, device=target_tensor.device) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            
            p_pol_pad = torch.nn.functional.pad(p_pol, (pad_left, pad_right), mode='replicate')
            s_pol_pad = torch.nn.functional.pad(s_pol, (pad_left, pad_right), mode='replicate')
            
            p_smoothed = torch.nn.functional.conv1d(p_pol_pad, kernel)
            s_smoothed = torch.nn.functional.conv1d(s_pol_pad, kernel)
            
            return torch.cat([p_smoothed.squeeze(1), s_smoothed.squeeze(1)], dim=-1)
            
        train_set.target = smooth_targets(train_set.target)
        val_set.target = smooth_targets(val_set.target)
    
    print(f"Datasets loaded: Train {len(train_set)} samples, Val {len(val_set)} samples in {time.time() - t0:.1f} s")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

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
        "dataset_prefixes": args.dataset_prefixes,
        "materials": args.materials,
        "target_key": args.target_key,
    }, ckpt_dir / "dataset_stats.pt")

    timings = {}
    all_history = {}

    if "mlp" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ForwardMLP")
        print("=" * 60)
        model_kwargs = dict(
            n_harmonics=n_harmonics, nx=128,
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=args.embed_dim, hidden_dims=tuple(args.mlp_hidden_dims),
            dropout=args.mlp_dropout
        )
        model = ForwardMLP(**model_kwargs)
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

        save_forward_checkpoint(model, hist, str(ckpt_dir / "forward_mlp.pt"), "ForwardMLP", model_kwargs, use_bfloat16=False)

    if "cnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SpatialCNN")
        print("=" * 60)
        model_kwargs = dict(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=args.embed_dim,
            grating_period=1000.0, conv_channels=tuple(args.cnn_conv_channels), kernel_size=args.cnn_kernel_size,
            fc_dims=tuple(args.cnn_fc_dims), dropout=args.cnn_dropout,
        )
        model = SpatialCNN(**model_kwargs)
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

        save_forward_checkpoint(model, hist, str(ckpt_dir / "spatial_cnn.pt"), "SpatialCNN", model_kwargs, use_bfloat16=False)

    if "skipcnn" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SkipCNN")
        print("=" * 60)
        model_kwargs = dict(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=args.embed_dim,
            grating_period=1000.0, conv_channels=tuple(args.skipcnn_conv_channels), kernel_size=args.skipcnn_kernel_size,
            fc_dims=tuple(args.skipcnn_fc_dims), dropout=args.skipcnn_dropout,
        )
        model = SkipCNN(**model_kwargs)
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

        save_forward_checkpoint(model, hist, str(ckpt_dir / "skip_cnn.pt"), "SkipCNN", model_kwargs, use_bfloat16=False)

    if "siren" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: SIREN")
        print("=" * 60)
        model_kwargs = dict(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=args.embed_dim,
            conv_channels=tuple(args.siren_conv_channels), kernel_size=args.siren_kernel_size, dropout=args.siren_dropout,
            siren_hidden=tuple(args.siren_fc_dims), latent_dim=args.siren_latent_dim, omega_0=args.siren_omega_0,
        )
        model = SIREN(**model_kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr / 10.0, patience=args.patience,
            device=device, use_bfloat16=False
        )
        elapsed = time.time() - t0
        timings["siren"] = elapsed
        all_history["siren"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_forward_checkpoint(model, hist, str(ckpt_dir / "siren.pt"), "SIREN", model_kwargs, use_bfloat16=True)

    if "transformer" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: TransformerForward")
        print("=" * 60)
        model_kwargs = dict(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=args.embed_dim,
            grating_period=1000.0, d_model=args.tf_d_model, nhead=args.tf_nhead, dim_feedforward=args.tf_dim_feedforward, num_layers=args.tf_num_layers, dropout=args.tf_dropout
        )
        model = TransformerForward(**model_kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        t0 = time.time()
        hist = train_forward_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr / 10.0, patience=args.patience,
            device=device, use_bfloat16=True
        )
        elapsed = time.time() - t0
        timings["transformer"] = elapsed
        all_history["transformer"] = hist
        print(f"  Time: {elapsed / 60:.1f} min")

        save_forward_checkpoint(model, hist, str(ckpt_dir / "transformer_forward.pt"), "TransformerForward", model_kwargs, use_bfloat16=True)

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
