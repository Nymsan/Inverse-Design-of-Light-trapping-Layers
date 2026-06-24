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
    train_tandem, train_cvae,
)
from Utils.checkpoint import (
    save_inverse_checkpoint, get_best_forward_model,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../Data")
    p.add_argument("--dataset_prefixes", nargs="+", default=["LHS_Dataset"])
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs for Tandem/Generative")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--latent_dim_gen", type=int, default=32)
    p.add_argument("--latent_dim_cvae", type=int, default=64)
    p.add_argument("--target_key", type=str, default="all_film")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--embed_dim", type=int, default=8)
    p.add_argument("--skip", nargs="*", default=[], choices=["tandem", "gen_tandem", "cvae"])
    p.add_argument("--force_forward_model", type=str, default=None, help="Force load a specific forward model (e.g. 'skip_cnn.pt')")
    
    # Architecture arguments
    # InverseDecoder
    p.add_argument("--inv_conv_channels", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--inv_kernel_size", type=int, default=7)
    p.add_argument("--inv_fc_dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--inv_dropout", type=float, default=0.05)
    
    # CVAE
    p.add_argument("--cvae_geo_enc_conv", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--cvae_geo_enc_kernel", type=int, default=7)
    p.add_argument("--cvae_geo_enc_fc", type=int, nargs="+", default=[256, 256])
    p.add_argument("--cvae_geo_enc_dropout", type=float, default=0.05)
    
    p.add_argument("--cvae_geo_dec_fc", type=int, nargs="+", default=[256, 256])
    p.add_argument("--cvae_geo_dec_dropout", type=float, default=0.05)
    
    p.add_argument("--cvae_spec_enc_conv", type=int, nargs="+", default=[32, 64, 128, 64])
    p.add_argument("--cvae_spec_enc_kernel", type=int, default=7)
    p.add_argument("--cvae_spec_enc_fc", type=int, nargs="+", default=[256, 256])
    p.add_argument("--cvae_spec_enc_dropout", type=float, default=0.05)
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
            device=device, forward_model=forward_model
        )
    else:
        hist = train_fn(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device
        )
    phases_trained.append("real")

    elapsed = time.time() - t0
    return hist, phases_trained, elapsed


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_name = "_".join(args.materials)
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
    
    print(f"Datasets loaded: Train {len(train_set)} samples, Val {len(val_set)} samples in {time.time() - t0:.1f} s")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

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

    # Find the best forward model (hardcoded to al_iter=0 to always use the clean base surrogate)
    forward_model, fwd_name, fwd_loss = get_best_forward_model(
        ckpt_dir, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_harmonics=n_harmonics, al_iter=0, force_model_name=args.force_forward_model
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
                n_materials=N_MATERIALS, latent_dim=0, fc_dims=tuple(args.inv_fc_dims),
                conv_channels=tuple(args.inv_conv_channels), kernel_size=args.inv_kernel_size, dropout=args.inv_dropout,
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
                n_materials=N_MATERIALS, latent_dim=latent_dim, fc_dims=tuple(args.inv_fc_dims),
                conv_channels=tuple(args.inv_conv_channels), kernel_size=args.inv_kernel_size, dropout=args.inv_dropout,
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
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=args.embed_dim,
            latent_dim=latent_dim, fc_dims=tuple(args.cvae_geo_enc_fc),
            conv_channels=tuple(args.cvae_geo_enc_conv), kernel_size=args.cvae_geo_enc_kernel, dropout=args.cvae_geo_enc_dropout,
        )
        geo_dec_kwargs = dict(
            latent_dim=latent_dim, n_geometry=n_continuous,
            n_materials=N_MATERIALS, geo_min=geo_min, geo_max=geo_max,
            hidden_dims=tuple(args.cvae_geo_dec_fc), dropout=args.cvae_geo_dec_dropout,
        )
        spec_enc_kwargs = dict(
            n_wavelengths=n_wavelengths, latent_dim=latent_dim,
            conv_channels=tuple(args.cvae_spec_enc_conv), kernel_size=args.cvae_spec_enc_kernel, 
            fc_dims=tuple(args.cvae_spec_enc_fc), dropout=args.cvae_spec_enc_dropout,
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
