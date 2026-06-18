#!/usr/bin/env python
"""
Train inverse models on generated grating data.
Automatically loads the best available frozen forward surrogate from checkpoints.

Training follows a 3-phase curriculum (each phase optional):
  Phase 1: Real curves (always runs)
  Phase 2: Wishful thinking — top quantile of curves interpolated toward 1.0
  Phase 3: Synthetic curves — Gaussians and step functions via frozen forward model

Usage:
    python Scripts/train_inverse.py --data_dir Data/LHS_Dataset_Si Data/LHS_Dataset_TiO2 Data/LHS_Dataset_Si3N4
    python Scripts/train_inverse.py --data_dir Data/LHS_Dataset_Si --skip_wishful --skip_synthetic
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
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE,
    GratingDataset,
    train_tandem, train_cvae, train_cvae_wishful,
)
from Utils.checkpoint import (
    save_inverse_checkpoint, get_best_forward_model,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", nargs="+", required=True)
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--wishful_epochs", type=int, default=100)
    p.add_argument("--synthetic_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--latent_dim_gen", type=int, default=32)
    p.add_argument("--latent_dim_cvae", type=int, default=64)
    p.add_argument("--target_key", type=str, default="all_film")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--skip", nargs="*", default=[], choices=["tandem", "gen_tandem", "cvae"])
    p.add_argument("--skip_wishful", action="store_true",
                    help="Skip Phase 2 (wishful thinking) training")
    p.add_argument("--skip_synthetic", action="store_true",
                    help="Skip Phase 3 (synthetic curves) training")
    return p.parse_args()


def _train_model_3phase(
    model, train_loader, val_loader, args, device, forward_model,
    train_fn, is_cvae=False, model_name="model",
):
    """Run the 3-phase training curriculum for an inverse model.
    
    Returns (history, phases_trained, elapsed_time).
    """
    phases_trained = []
    t0 = time.time()

    # --- Phase 1: Real curves (always runs) ---
    print("  Phase 1: Real curves")
    if is_cvae:
        hist = train_fn(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, synthetic_phase=False,
        )
    else:
        hist = train_fn(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, synthetic_phase=False,
        )
    phases_trained.append("real")

    # --- Phase 2: Wishful thinking (optional, default on) ---
    if not args.skip_wishful and args.wishful_epochs > 0:
        if is_cvae:
            print(f"\n  Phase 2: Wishful thinking ({args.wishful_epochs} epochs)")
            hist_wishful = train_cvae_wishful(
                model, train_loader, val_loader,
                epochs=args.wishful_epochs, lr=args.lr * 0.1,
                patience=min(args.patience, 50),
                device=device,
            )
            for k in hist:
                hist[k].extend(hist_wishful.get(k, []))
            phases_trained.append("wishful")
        else:
            print(f"\n  Phase 2: Wishful thinking — skipped for {model_name} (only supported for CVAE)")
    elif args.skip_wishful:
        print("\n  Phase 2: Wishful thinking — SKIPPED (--skip_wishful)")

    # --- Phase 3: Synthetic curves (optional, default on) ---
    if not args.skip_synthetic and args.synthetic_epochs > 0:
        if is_cvae:
            if forward_model is not None:
                print(f"\n  Phase 3: Synthetic curves ({args.synthetic_epochs} epochs)")
                hist_synth = train_fn(
                    model, train_loader, val_loader,
                    epochs=args.synthetic_epochs, lr=args.lr, patience=args.patience,
                    device=device, forward_model=forward_model, synthetic_phase=True,
                )
                for k in hist:
                    hist[k].extend(hist_synth.get(k, []))
                phases_trained.append("synthetic")
            else:
                print("\n  WARNING: Skipping Phase 3 for CVAE as forward_model is missing.")
        else:
            print(f"\n  Phase 3: Synthetic curves ({args.synthetic_epochs} epochs)")
            hist_synth = train_fn(
                model, train_loader, val_loader,
                epochs=args.synthetic_epochs, lr=args.lr, patience=args.patience,
                device=device, synthetic_phase=True,
            )
            for k in hist:
                hist[k].extend(hist_synth.get(k, []))
            phases_trained.append("synthetic")
    elif args.skip_synthetic:
        print("\n  Phase 3: Synthetic curves — SKIPPED (--skip_synthetic)")

    elapsed = time.time() - t0
    return hist, phases_trained, elapsed


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
        geo_min = train_set.geometry.min(dim=0).values
        geo_max = train_set.geometry.max(dim=0).values

    # Find the best forward model
    forward_model, fwd_name, fwd_loss = get_best_forward_model(
        ckpt_dir, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_harmonics=n_harmonics
    )
    
    if forward_model is not None:
        print(f"\n=> Loaded BEST forward model: {fwd_name} (val_loss = {fwd_loss:.6f})")
    else:
        print("\n=> WARNING: No forward models found in Checkpoints directory! Tandem training will be skipped.")

    timings = {}
    all_history = {}

    # ==================== TandemNetwork ====================
    if "tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: TandemNetwork")
        print("=" * 60)
        if forward_model is None:
            print("  SKIPPED — no forward model available.")
        else:
            decoder_kwargs = dict(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=0,
                geo_min=geo_min, geo_max=geo_max,
            )
            decoder = InverseDecoder(**decoder_kwargs)
            tandem = TandemNetwork(inverse_decoder=decoder, forward_model=forward_model)
            n_params = sum(p.numel() for p in tandem.inverse_decoder.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
            print(f"  Trainable parameters (decoder only): {n_params:,}")

            hist, phases, elapsed = _train_model_3phase(
                tandem, train_loader, val_loader, args, device, forward_model,
                train_fn=train_tandem, is_cvae=False, model_name="TandemNetwork",
            )
            
            timings["tandem"] = elapsed
            all_history["tandem"] = hist
            print(f"  Time: {elapsed / 60:.1f} min  |  Phases: {phases}")

            # Build config for checkpoint (exclude non-serializable tensors)
            model_config = {
                "inverse_decoder": {
                    k: v for k, v in decoder_kwargs.items()
                    if k not in ("geo_min", "geo_max")
                },
            }
            save_inverse_checkpoint(
                tandem, hist, str(ckpt_dir / "tandem.pt"),
                model_class_name="TandemNetwork",
                model_config=model_config,
                forward_model_name=fwd_name,
                phases_trained=phases,
            )

    # ==================== GenerativeTandemNetwork ====================
    if "gen_tandem" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: GenerativeTandemNetwork")
        print("=" * 60)
        if forward_model is None:
            print("  SKIPPED — no forward model available.")
        else:
            latent_dim = args.latent_dim_gen
            decoder_kwargs = dict(
                n_wavelengths=n_wavelengths, n_geometry=n_continuous,
                n_materials=N_MATERIALS, latent_dim=latent_dim,
                geo_min=geo_min, geo_max=geo_max,
            )
            decoder = InverseDecoder(**decoder_kwargs)
            gen_tandem = GenerativeTandemNetwork(
                inverse_decoder=decoder, forward_model=forward_model,
                latent_dim=latent_dim,
            )
            n_params = sum(p.numel() for p in gen_tandem.inverse_decoder.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
            print(f"  Trainable parameters (decoder only): {n_params:,}")

            hist, phases, elapsed = _train_model_3phase(
                gen_tandem, train_loader, val_loader, args, device, forward_model,
                train_fn=train_tandem, is_cvae=False, model_name="GenerativeTandemNetwork",
            )

            timings["generative_tandem"] = elapsed
            all_history["generative_tandem"] = hist
            print(f"  Time: {elapsed / 60:.1f} min  |  Phases: {phases}")

            model_config = {
                "inverse_decoder": {
                    k: v for k, v in decoder_kwargs.items()
                    if k not in ("geo_min", "geo_max")
                },
                "latent_dim": latent_dim,
            }
            save_inverse_checkpoint(
                gen_tandem, hist, str(ckpt_dir / "generative_tandem.pt"),
                model_class_name="GenerativeTandemNetwork",
                model_config=model_config,
                forward_model_name=fwd_name,
                phases_trained=phases,
            )

    # ==================== ContrastiveVAE ====================
    if "cvae" not in args.skip:
        print("\n" + "=" * 60)
        print("Training: ContrastiveVAE")
        print("=" * 60)
        latent_dim = args.latent_dim_cvae
        
        geo_enc_kwargs = dict(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=latent_dim, fc_dims=(256, 256),
        )
        geo_dec_kwargs = dict(
            latent_dim=latent_dim, n_geometry=n_continuous,
            n_materials=N_MATERIALS, geo_min=geo_min, geo_max=geo_max,
            hidden_dims=(256, 256),
        )
        spec_enc_kwargs = dict(
            n_wavelengths=n_wavelengths, latent_dim=latent_dim,
            conv_channels=(32, 64, 128, 64), fc_dims=(256, 256),
        )
        cvae_kwargs = dict(
            margin_radius=1.0, beta=1e-3, gamma=1.0,
        )

        geo_enc = GeometryEncoder(**geo_enc_kwargs)
        geo_dec = GeometryDecoder(**geo_dec_kwargs)
        spec_enc = SpectrumEncoder(**spec_enc_kwargs)
        cvae = ContrastiveVAE(
            geometry_encoder=geo_enc, geometry_decoder=geo_dec,
            spectrum_encoder=spec_enc, **cvae_kwargs,
        )
        n_geo_enc = sum(p.numel() for p in geo_enc.parameters())
        n_geo_dec = sum(p.numel() for p in geo_dec.parameters())
        n_spec_enc = sum(p.numel() for p in spec_enc.parameters())
        n_cvae_total = sum(p.numel() for p in cvae.parameters())
# Removed torch.compile due to dynamic tau parameter causing recompilation hangs
        print(f"  Parameters: Total={n_cvae_total:,} | GeoEnc={n_geo_enc:,} | GeoDec={n_geo_dec:,} | SpecEnc={n_spec_enc:,}")

        hist, phases, elapsed = _train_model_3phase(
            cvae, train_loader, val_loader, args, device, forward_model,
            train_fn=train_cvae, is_cvae=True, model_name="ContrastiveVAE",
        )

        timings["cvae"] = elapsed
        all_history["cvae"] = hist
        print(f"  Time: {elapsed / 60:.1f} min  |  Phases: {phases}")

        model_config = {
            "geometry_encoder": {
                k: v for k, v in geo_enc_kwargs.items()
                if k not in ("geo_min", "geo_max")
            },
            "geometry_decoder": {
                k: v for k, v in geo_dec_kwargs.items()
                if k not in ("geo_min", "geo_max")
            },
            "spectrum_encoder": {
                k: v for k, v in spec_enc_kwargs.items()
            },
            **cvae_kwargs,
        }
        save_inverse_checkpoint(
            cvae, hist, str(ckpt_dir / "cvae.pt"),
            model_class_name="ContrastiveVAE",
            model_config=model_config,
            forward_model_name=fwd_name,
            phases_trained=phases,
        )

    # ==================== Summary ====================
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
