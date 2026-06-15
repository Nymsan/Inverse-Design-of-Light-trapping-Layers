#!/usr/bin/env python
"""
Train inverse models on generated grating data.
Automatically loads the best available frozen forward surrogate from checkpoints.

Usage:
    python Scripts/train_inverse.py --data_dir Data/LHS_Dataset_Si
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import glob
import random
from pathlib import Path

import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader, random_split

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS,
    ForwardMLP, SpatialCNN, SkipCNN, SIREN,
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE,
    GratingDataset,
    train_tandem, train_cvae, train_cvae_wishful,
)

def save_checkpoint(model, history: dict, path: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
    }, path)

def get_best_forward_model(ckpt_dir, n_continuous, n_wavelengths, n_harmonics):
    best_loss = float('inf')
    best_name = None
    best_model = None

    def _clean_state_dict(sd):
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("_orig_mod."):
                clean_sd[k[len("_orig_mod."):]] = v
            else:
                clean_sd[k] = v
        return clean_sd

    # ForwardMLP
    p = ckpt_dir / "forward_mlp.pt"
    if p.exists():
        model = ForwardMLP(
            n_harmonics=n_harmonics, nx=128,
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            
        )
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            val_loss = min(hist["val_loss"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = "forward_mlp"
                model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]), strict=True)
                best_model = model

    # SpatialCNN
    p = ckpt_dir / "spatial_cnn.pt"
    if p.exists():
        model = SpatialCNN(
            n_harmonics=n_harmonics, nx=128,
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, 
        )
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            val_loss = min(hist["val_loss"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = "spatial_cnn"
                model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]), strict=True)
                best_model = model

    # SkipCNN
    p = ckpt_dir / "skip_cnn.pt"
    if p.exists():
        model = SkipCNN(
            n_harmonics=n_harmonics, nx=128,
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            grating_period=1000.0, 
        )
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            val_loss = min(hist["val_loss"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = "skip_cnn"
                model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]), strict=True)
                best_model = model

    # SIREN
    p = ckpt_dir / "siren.pt"
    if p.exists():
        model = SIREN(
            n_harmonics=n_harmonics, nx=128, n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            conv_channels=(32, 64, 128, 64), kernel_size=7, dropout=0.05, siren_hidden=(256, 256, 256), latent_dim=128, omega_0=10.0
        )
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            val_loss = min(hist["val_loss"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = "siren"
                model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]), strict=True)
                best_model = model



    return best_model, best_name, best_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", nargs="+", required=True)
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--latent_dim_gen", type=int, default=32)
    p.add_argument("--latent_dim_cvae", type=int, default=64)
    p.add_argument("--target_key", type=str, default="all_film")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--skip", nargs="*", default=[], choices=["tandem", "gen_tandem", "cvae"])
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
    
    stats_path = ckpt_dir / "dataset_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        geo_min = stats["geo_min"]
        geo_max = stats["geo_max"]
    else:
        geo_min = dataset.geometry.min(dim=0).values
        geo_max = dataset.geometry.max(dim=0).values

    # Find the best forward model
    forward_model, fwd_name, fwd_loss = get_best_forward_model(ckpt_dir, n_continuous, n_wavelengths, n_harmonics)
    
    if forward_model is not None:
        print(f"\n=> Loaded BEST forward model: {fwd_name} (val_loss = {fwd_loss:.6f})")
    else:
        print("\n=> WARNING: No forward models found in Checkpoints directory! Tandem training will be skipped.")

    timings = {}
    all_history = {}

    if "tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: TandemNetwork")
        print("=" * 60)
        if forward_model is None:
            print("  SKIPPED — no forward model available.")
        else:
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=0,
                geo_min=geo_min, geo_max=geo_max,
                
            )
            tandem = TandemNetwork(inverse_decoder=decoder, forward_model=forward_model)
            n_params = sum(p.numel() for p in tandem.inverse_decoder.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
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

    if "gen_tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: GenerativeTandemNetwork")
        print("=" * 60)
        if forward_model is None:
            print("  SKIPPED — no forward model available.")
        else:
            latent_dim = args.latent_dim_gen
            decoder = InverseDecoder(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=latent_dim,
                geo_min=geo_min, geo_max=geo_max,
                
            )
            gen_tandem = GenerativeTandemNetwork(
                inverse_decoder=decoder, forward_model=forward_model,
                latent_dim=latent_dim,
            )
            n_params = sum(p.numel() for p in gen_tandem.inverse_decoder.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
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

    if "cvae" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ContrastiveVAE")
        print("=" * 60)
        latent_dim = args.latent_dim_cvae
        geo_enc = GeometryEncoder(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=latent_dim, fc_dims=(256, 256),
        )
        geo_dec = GeometryDecoder(
            latent_dim=latent_dim, n_geometry=n_continuous,
            n_materials=N_MATERIALS, geo_min=geo_min, geo_max=geo_max,
            hidden_dims=(256, 256),
        )
        spec_enc = SpectrumEncoder(
            n_wavelengths=n_wavelengths, latent_dim=latent_dim,
            conv_channels=(32, 64, 128, 64), fc_dims=(256, 256),
        )
        cvae = ContrastiveVAE(
            geometry_encoder=geo_enc, geometry_decoder=geo_dec,
            spectrum_encoder=spec_enc, margin_radius=1.0,
            beta=1e-3, gamma=1.0,
        )
        n_params = sum(p.numel() for p in cvae.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
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

        print("\n" + "-" * 60)
        print("Wishful Finetuning: CVAE")
        print("-" * 60)
        t0 = time.time()
        # Finetune with lower lr and fewer epochs
        hist_ft = train_cvae_wishful(
            cvae, train_loader, val_loader,
            epochs=args.epochs // 2 if args.epochs > 1 else 1, lr=args.lr * 0.1, patience=args.patience // 2 if args.patience > 1 else 1,
            device=device,
        )
        elapsed = time.time() - t0
        timings["cvae_wishful"] = elapsed
        all_history["cvae_wishful"] = hist_ft
        print(f"  Time: {elapsed / 60:.1f} min")
        save_checkpoint(cvae, hist_ft, str(ckpt_dir / "cvae_wishful.pt"))

    history_path = ckpt_dir / "inverse_history.json"
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
