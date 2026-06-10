#!/usr/bin/env python
"""
Evaluate trained inverse models and generate performance report.

Usage:
    python Scripts/evaluate_inverse.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS,
    ForwardMLP, SpatialCNN,
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE,
    GratingDataset,
)

from Scripts.train_inverse import get_best_forward_model

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

WAVELENGTHS = np.linspace(300, 1100, 161)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    p.add_argument("--n_eval", type=int, default=2000, help="Max samples for evaluation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_loss_curves(all_history: dict, save_path: str):
    if not all_history:
        return
    n_models = len(all_history)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, hist) in zip(axes, all_history.items()):
        if "train_loss" not in hist:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], label="Train", alpha=0.8)
        ax.semilogy(epochs, hist["val_loss"], label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Inverse Training Loss Curves", fontsize=15, y=1.02)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_inverse_performance(inverse_models, forward_model, val_loader, save_path: str, n_wavelengths: int):
    if not inverse_models:
        return
    batch = next(iter(val_loader))
    target = batch["target"]
    n_show = min(4, target.shape[0])
    n_wl_half = n_wavelengths // 2

    fig, axes = plt.subplots(n_show, 2, figsize=(10, 3 * n_show), squeeze=False, layout="constrained")
    colors = plt.cm.tab10(np.linspace(0, 1, len(inverse_models)))

    for i in range(n_show):
        curve = target[i:i+1]
        preds = {}
        for name, inv_model in inverse_models.items():
            if name == "tandem":
                pred_geo, mat_oh, _ = inv_model.inverse_decoder(curve, tau=0.1)
                preds[name] = _run_forward(forward_model, pred_geo, mat_oh)
            elif name == "generative_tandem":
                designs = inv_model.sample_diverse_designs(curve, n_samples=1, tau=0.1)
                preds[name] = _run_forward(forward_model, designs["pred_geometry"], designs["material_onehot"])
            elif name == "cvae":
                z_y = inv_model.spectrum_encoder(curve)
                pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_y, tau=0.1, hard=True)
                preds[name] = _run_forward(forward_model, pred_geo, mat_oh)

        for pol_idx, pol_label in enumerate(["p-pol", "s-pol"]):
            ax = axes[i, pol_idx]
            start = pol_idx * n_wl_half
            end = start + n_wl_half
            ax.plot(WAVELENGTHS, curve[0, start:end].numpy(), "k-", lw=2.5, label="Target", zorder=10)
            
            for (name, pred), c in zip(preds.items(), colors):
                ax.plot(WAVELENGTHS, pred[0, start:end].numpy(),
                        "--", color=c, lw=2.0, label=name.replace("_", " ").title())
                
            ax.set_xlim(300, 1100)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Absorptance")
            if i == 0:
                ax.set_title(f"Target vs. Obtained ({pol_label})")
            if i == 0 and pol_idx == 0:
                ax.legend(fontsize=9)
            if i == n_show - 1:
                ax.set_xlabel("Wavelength (nm)")

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def _run_forward(forward_model, pred_geo, mat_oh):
    if hasattr(forward_model, "_build_profile"): # CNN
        return forward_model(pred_geo[:, :-1].view(pred_geo.shape[0], -1, 2), pred_geo[:, -1:], mat_oh.argmax(dim=-1))
    else:
        return forward_model(pred_geo, mat_oh.argmax(dim=-1))


@torch.no_grad()
def plot_candidate_designs(inv_model, forward_model, val_loader, save_path: str, n_wavelengths: int, stats: dict):
    batch = next(iter(val_loader))
    curve = batch["target"][0:1] # (1, N)
    
    n_samples = 5
    if hasattr(inv_model, "sample_diverse_designs"):
        # Generative Tandem
        designs = inv_model.sample_diverse_designs(curve, n_samples=n_samples, tau=0.1)
        pred_geo = designs["pred_geometry"] # (5, 12)
        mat_oh = designs["material_onehot"] # (5, 3)
    elif hasattr(inv_model, "spectrum_encoder"):
        # Contrastive VAE: Add noise to z_y to sample the latent space
        z_y = inv_model.spectrum_encoder(curve)
        z_noisy = z_y.expand(n_samples, -1) + torch.randn(n_samples, z_y.shape[1], device=z_y.device) * 0.5
        pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_noisy, tau=0.1, hard=True)
    else:
        return

    preds = _run_forward(forward_model, pred_geo, mat_oh)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples), squeeze=False, layout="constrained")
    n_wl_half = n_wavelengths // 2
    mat_names = list(stats["materials"].keys())
    
    # Pre-build coordinates for the profile
    n_harmonics = stats["n_harmonics"]
    r_grid = np.linspace(0, 1000.0, 256)
    harmonic_idx = np.arange(1, n_harmonics + 1)
    
    for i in range(n_samples):
        # Left column: Curves
        ax = axes[i, 0]
        ax.plot(WAVELENGTHS, curve[0, :n_wl_half].numpy(), "k-", lw=2.5, label="Target p-pol", zorder=10)
        ax.plot(WAVELENGTHS, preds[i, :n_wl_half].numpy(), "r--", lw=2.0, label="Pred p-pol")
        ax.plot(WAVELENGTHS, curve[0, n_wl_half:].numpy(), "k:", lw=2.5, label="Target s-pol", zorder=10)
        ax.plot(WAVELENGTHS, preds[i, n_wl_half:].numpy(), "b--", lw=2.0, label="Pred s-pol")
        ax.set_xlim(300, 1100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Absorptance")
        if i == 0:
            ax.set_title("Target vs Candidate Spectra")
            ax.legend(fontsize=8, ncol=2)
            
        # Right column: Geometry profile
        ax = axes[i, 1]
        mat_idx = mat_oh[i].argmax().item()
        mat_name = mat_names[mat_idx]
        h = pred_geo[i, -1].item()
        
        # Build 1D profile manually
        px = pred_geo[i, :-1].view(-1, 2)
        amps = px[:, 0].numpy()
        phases = px[:, 1].numpy()
        
        grating_height = 2.0 * amps.sum() + 1e-9
        arg = 2.0 * np.pi * harmonic_idx[:, None] * r_grid[None, :] / 1000.0 - phases[:, None]
        cosines = amps[:, None] * np.cos(arg)
        prof = grating_height / 2.0 + cosines.sum(axis=0)
        p_min = prof.min()
        p_max = prof.max()
        prof = (prof - p_min) / (p_max - p_min + 1e-9)
        
        # Denormalize geometry to nm
        geo_min = stats["geo_min"].numpy()
        geo_max = stats["geo_max"].numpy()
        h_nm = h * (geo_max[-1] - geo_min[-1]) + geo_min[-1]
        
        ax.fill_between(r_grid, 0, prof * h_nm, color="gray", alpha=0.5)
        ax.set_ylim(0, max(600, h_nm * 1.2))
        ax.set_xlim(0, 1000)
        ax.set_title(f"Candidate {i+1} ({mat_name}, h={h_nm:.1f}nm)")
        ax.set_ylabel("Thickness (nm)")
        if i == n_samples - 1:
            ax.set_xlabel("x (nm)")
            
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_unit_absorptance(inverse_models, forward_model, save_path: str, n_wavelengths: int, stats: dict):
    curve = torch.ones(1, n_wavelengths)
    n_wl_half = n_wavelengths // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False, layout="constrained")
    colors = plt.cm.tab10(np.linspace(0, 1, len(inverse_models)))
    mat_names = list(stats["materials"].keys())

    obtained_curves = {}
    
    for name, inv_model in inverse_models.items():
        if name == "tandem":
            pred_geo, mat_oh, _ = inv_model.inverse_decoder(curve, tau=0.1)
            obtained_curves[name] = _run_forward(forward_model, pred_geo, mat_oh)
        elif name == "generative_tandem":
            designs = inv_model.sample_diverse_designs(curve, n_samples=1, tau=0.1)
            obtained_curves[name] = _run_forward(forward_model, designs["pred_geometry"], designs["material_onehot"])
        elif name in ("cvae", "cvae_wishful"):
            z_y = inv_model.spectrum_encoder(curve)
            pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_y, tau=0.1, hard=True)
            obtained_curves[name] = _run_forward(forward_model, pred_geo, mat_oh)

    for pol_idx, pol_label in enumerate(["p-pol", "s-pol"]):
        ax = axes[0, pol_idx]
        start = pol_idx * n_wl_half
        end = start + n_wl_half
        
        ax.plot(WAVELENGTHS, curve[0, start:end].numpy(), "k-", lw=2.5, label="Target (100%)", zorder=10)
        
        for (name, pred), c in zip(obtained_curves.items(), colors):
            ax.plot(WAVELENGTHS, pred[0, start:end].numpy(), "--", color=c, lw=2.0, label=name.replace("_", " ").title())
            
        ax.set_xlim(300, 1100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorptance")
        ax.set_title(f"Perfect Absorber Challenge ({pol_label})")
        if pol_idx == 0:
            ax.legend()
            
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    mat_dirs = {k: str(PROJECT_ROOT / "Data" / Path(v).name) for k, v in stats["materials"].items()}
    target_key = stats["target_key"]
    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={list(mat_dirs.keys())}")

    full_dataset = GratingDataset(
        data_dirs=mat_dirs, target_key=target_key,
        geo_min=stats["geo_min"], geo_max=stats["geo_max"],
    )
    n_val = int(len(full_dataset) * 0.15)
    n_train = len(full_dataset) - n_val
    _, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    if args.n_eval < len(val_ds):
        val_ds, _ = random_split(val_ds, [args.n_eval, len(val_ds) - args.n_eval],
                                 generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    print(f"Validation samples: {len(val_ds)}")

    forward_model, fwd_name, fwd_loss = get_best_forward_model(ckpt_dir, n_continuous, n_wavelengths, n_harmonics)
    if forward_model is not None:
        forward_model.eval()
        print(f"\n=> Loaded BEST forward model for evaluation: {fwd_name} (val_loss = {fwd_loss:.6f})")
    else:
        print("\n=> ERROR: No forward models found. Run train_forward.py first!")
        return

    all_history = {}
    inverse_models = {}
    
    for name in ("tandem", "generative_tandem", "cvae", "cvae_wishful"):
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            all_history[name] = ckpt.get("history", {})

    tandem_path = ckpt_dir / "tandem.pt"
    if tandem_path.exists():
        dec = InverseDecoder(
            n_wavelengths=n_wavelengths, n_geometry=n_continuous,
            n_materials=N_MATERIALS, latent_dim=0,
            hidden_dims=(256, 512, 512, 256),
        )
        tandem = TandemNetwork(inverse_decoder=dec, forward_model=forward_model)
        ckpt = torch.load(tandem_path, map_location="cpu", weights_only=False)
        tandem.load_state_dict(ckpt["model_state_dict"])
        tandem.eval()
        inverse_models["tandem"] = tandem
        print("Loaded: tandem")

    gen_path = ckpt_dir / "generative_tandem.pt"
    if gen_path.exists():
        dec = InverseDecoder(
            n_wavelengths=n_wavelengths, n_geometry=n_continuous,
            n_materials=N_MATERIALS, latent_dim=32,
            hidden_dims=(256, 512, 512, 256),
        )
        gen_tandem = GenerativeTandemNetwork(inverse_decoder=dec, forward_model=forward_model, latent_dim=32)
        ckpt = torch.load(gen_path, map_location="cpu", weights_only=False)
        gen_tandem.load_state_dict(ckpt["model_state_dict"])
        gen_tandem.eval()
        inverse_models["generative_tandem"] = gen_tandem
        print("Loaded: generative_tandem")
        
    cvae_path = ckpt_dir / "cvae.pt"
    if cvae_path.exists():
        geo_enc = GeometryEncoder(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=64, hidden_dims=(256, 256),
        )
        geo_dec = GeometryDecoder(
            latent_dim=64, n_geometry=n_continuous,
            n_materials=N_MATERIALS, hidden_dims=(256, 256),
        )
        spec_enc = SpectrumEncoder(
            n_wavelengths=n_wavelengths, latent_dim=64,
            hidden_dims=(128, 256, 128),
        )
        cvae = ContrastiveVAE(
            geometry_encoder=geo_enc, geometry_decoder=geo_dec,
            spectrum_encoder=spec_enc, margin_radius=1.0,
            beta=1e-3, gamma=1.0,
        )
        ckpt = torch.load(cvae_path, map_location="cpu", weights_only=False)
        cvae.load_state_dict(ckpt["model_state_dict"])
        cvae.eval()
        inverse_models["cvae"] = cvae
        print("Loaded: cvae")
        
    cvae_wishful_path = ckpt_dir / "cvae_wishful.pt"
    if cvae_wishful_path.exists():
        geo_enc = GeometryEncoder(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=64, hidden_dims=(256, 256),
        )
        geo_dec = GeometryDecoder(
            latent_dim=64, n_geometry=n_continuous,
            n_materials=N_MATERIALS, hidden_dims=(256, 256),
        )
        spec_enc = SpectrumEncoder(
            n_wavelengths=n_wavelengths, latent_dim=64,
            hidden_dims=(128, 256, 128),
        )
        cvae_wishful = ContrastiveVAE(
            geometry_encoder=geo_enc, geometry_decoder=geo_dec,
            spectrum_encoder=spec_enc, margin_radius=1.0,
            beta=1e-3, gamma=1.0,
        )
        ckpt = torch.load(cvae_wishful_path, map_location="cpu", weights_only=False)
        cvae_wishful.load_state_dict(ckpt["model_state_dict"])
        cvae_wishful.eval()
        inverse_models["cvae_wishful"] = cvae_wishful
        print("Loaded: cvae_wishful")

    print("\nGenerating evaluation plots...")
    if all_history:
        plot_loss_curves(all_history, str(eval_dir / "inverse_loss_curves.png"))

    if inverse_models:
        plot_inverse_performance(inverse_models, forward_model, val_loader,
                                str(eval_dir / "inverse_performance.png"), n_wavelengths)
        
        if "generative_tandem" in inverse_models:
            plot_candidate_designs(inverse_models["generative_tandem"], forward_model, val_loader,
                                   str(eval_dir / "candidate_designs_gen_tandem.png"), n_wavelengths, stats)
        
        if "cvae" in inverse_models:
            plot_candidate_designs(inverse_models["cvae"], forward_model, val_loader,
                                   str(eval_dir / "candidate_designs_cvae.png"), n_wavelengths, stats)
                                   
        if "cvae_wishful" in inverse_models:
            plot_candidate_designs(inverse_models["cvae_wishful"], forward_model, val_loader,
                                   str(eval_dir / "candidate_designs_cvae_wishful.png"), n_wavelengths, stats)
            
        plot_unit_absorptance(inverse_models, forward_model,
                              str(eval_dir / "unit_absorptance.png"), n_wavelengths, stats)

    print(f"\nAll evaluation outputs saved to: {eval_dir}")

if __name__ == "__main__":
    main()
