#!/usr/bin/env python
"""
Evaluate trained surrogate models and generate performance report.

Usage:
    python Scripts/evaluate_models.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4

Generates:
    Checkpoints/<run>/evaluation/
        loss_curves.png
        forward_parity.png
        spectrum_samples.png
        inverse_diversity.png
        metrics.json
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
    ForwardMLP, SpatialCNN, SpatialMamba,
    InverseDecoder, TandemNetwork, GenerativeTandemNetwork,
    GeometryEncoder, GeometryDecoder, SpectrumEncoder, ContrastiveVAE,
    GratingDataset,
)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

WAVELENGTHS = np.linspace(300, 1100, 161)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt.get("history", {}), model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    p.add_argument("--n_eval", type=int, default=2000, help="Max samples for evaluation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Plot 1: Training loss curves ──
def plot_loss_curves(all_history: dict, save_path: str):
    n_models = len(all_history)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, hist) in zip(axes, all_history.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], label="Train", alpha=0.8)
        ax.semilogy(epochs, hist["val_loss"], label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Loss Curves", fontsize=15, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 2: Forward model parity plot ──
@torch.no_grad()
def plot_forward_parity(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int):
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, model) in zip(axes, models.items()):
        all_pred, all_true = [], []
        for batch in val_loader:
            geo, px, mat, target = (batch["geometry"], batch["params_x"],
                                    batch["material_id"], batch["target"])
            if "cnn" in name.lower():
                h_val = geo[:, -1:]
                pred = model(px, h_val, mat)
            else:
                pred = model(geo, mat)
            all_pred.append(pred)
            all_true.append(target)

        pred = torch.cat(all_pred).numpy().flatten()
        true = torch.cat(all_true).numpy().flatten()

        ax.hexbin(true, pred, gridsize=80, cmap="inferno", mincnt=1)
        ax.plot([0, 1], [0, 1], "w--", lw=1.5, alpha=0.8)
        ax.set_xlabel("True Absorptance")
        ax.set_ylabel("Predicted Absorptance")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        if name == list(models.keys())[0]:
            ax.set_title(f"Parity: {name.replace('_', ' ').title()}", fontsize=12)
        else:
            ax.set_title(name.replace('_', ' ').title(), fontsize=12)

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 3: Example spectrum predictions ──
@torch.no_grad()
def plot_spectrum_samples(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int):
    batch = next(iter(val_loader))
    geo, px, mat, target = batch["geometry"], batch["params_x"], batch["material_id"], batch["target"]
    n_samples = min(6, target.shape[0])
    n_wl_half = n_wavelengths // 2

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples), squeeze=False, layout="constrained")
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i in range(n_samples):
        preds = {}
        for name, model in models.items():
            if "cnn" in name.lower():
                pred = model(px[i:i+1], geo[i:i+1, -1:], mat[i:i+1])
            else:
                pred = model(geo[i:i+1], mat[i:i+1])
            preds[name] = pred

        for pol_idx, pol_label in enumerate(["p-pol", "s-pol"]):
            ax = axes[i, pol_idx]
            start = pol_idx * n_wl_half
            end = start + n_wl_half
            ax.plot(WAVELENGTHS, target[i, start:end].numpy(), "k-", lw=2.5, label="Truth", alpha=0.5)

            for (name, pred), c in zip(preds.items(), colors):
                ax.plot(WAVELENGTHS, pred[0, start:end].numpy(),
                        "--", color=c, lw=2.0, label=name.replace("_", " ").title())
            
            ax.set_xlim(300, 1100)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Absorptance")
            if i == 0:
                ax.set_title(f"Predictions ({pol_label})")
            if i == 0 and pol_idx == 0:
                ax.legend(fontsize=9)
            if i == n_samples - 1:
                ax.set_xlabel("Wavelength (nm)")

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 4: Inverse model performance ──
@torch.no_grad()
def plot_inverse_performance(inverse_models, forward_model, val_loader, save_path: str, n_wavelengths: int):
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
                preds[name] = forward_model(pred_geo, mat_oh)
            elif name == "generative_tandem":
                designs = inv_model.sample_diverse_designs(curve, n_samples=1, tau=0.1)
                preds[name] = forward_model(designs["pred_geometry"], designs["material_onehot"])
            elif name == "cvae":
                z_y = inv_model.spectrum_encoder(curve)
                pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_y, tau=0.1, hard=True)
                preds[name] = forward_model(pred_geo, mat_oh)

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


# ── Plot 5: Unit absorptance test ──
from Utils.utils import RCWAConfig, get_absorptance_curve

@torch.no_grad()
def plot_unit_absorptance(inverse_models, forward_model, save_path: str, n_wavelengths: int, stats: dict):
    # Request 100% absorptance curve
    curve = torch.ones(1, n_wavelengths)
    n_wl_half = n_wavelengths // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False, layout="constrained")
    colors = plt.cm.tab10(np.linspace(0, 1, len(inverse_models)))
    mat_names = list(stats["materials"].keys())

    obtained_curves = {}
    rcwa_curves = {}
    
    print("  Simulating proposed designs via RCWA (this may take a minute)...")
    for name, inv_model in inverse_models.items():
        if name == "tandem":
            pred_geo, mat_oh, _ = inv_model.inverse_decoder(curve, tau=0.1)
        elif name == "generative_tandem":
            designs = inv_model.sample_diverse_designs(curve, n_samples=1, tau=0.1)
            pred_geo, mat_oh = designs["pred_geometry"], designs["material_onehot"]
        elif name == "cvae":
            z_y = inv_model.spectrum_encoder(curve)
            pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_y, tau=0.1, hard=True)

        obtained_curves[name] = forward_model(pred_geo, mat_oh)

        # Denormalize to get real physical parameters
        geo_unnorm = pred_geo * (stats["geo_max"] - stats["geo_min"]) + stats["geo_min"]
        
        h_val = geo_unnorm[0, -2].item()
        params_x = geo_unnorm[0, :-2].reshape(5, 2)
        mat_name = mat_names[mat_oh.argmax(dim=-1).item()]

        config = RCWAConfig(grating_material=mat_name, h=h_val, inc_ang=1e-3 * np.pi/180, azi_ang=1e-3 * np.pi/180)
        
        print(f"  -> Torcwa RCWA for {name}: {mat_name}, h={h_val:.1f}nm")
        A_film, _ = get_absorptance_curve(
            params_x=params_x, params_y=None,
            wavelengths=torch.tensor(WAVELENGTHS, dtype=torch.float32) + 1e-3,
            config=config, show_progress=False
        )
        rcwa_curves[name] = torch.cat([A_film[:, 0], A_film[:, 1]], dim=-1)

    for pol_idx, pol_label in enumerate(["p-pol", "s-pol"]):
        ax = axes[0, pol_idx]
        start = pol_idx * n_wl_half
        end = start + n_wl_half
        
        ax.plot(WAVELENGTHS, curve[0, start:end].numpy(),
                "k-", lw=2.5, label="Target (1.0)", zorder=10)
        
        for (name, pred), c in zip(obtained_curves.items(), colors):
            ax.plot(WAVELENGTHS, pred[0, start:end].numpy(),
                    "--", color=c, lw=1.5, label=f"{name.replace('_', ' ').title()} (Surrogate)")
            ax.plot(WAVELENGTHS, rcwa_curves[name][start:end].numpy(),
                    "-", color=c, lw=1.5, label=f"{name.replace('_', ' ').title()} (True RCWA)")
            
        ax.set_xlim(300, 1100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Absorptance")
        ax.set_title(f"Holy Grail: 100% Absorptance ({pol_label})")
        if pol_idx == 0:
            ax.legend(fontsize=7, ncol=1)
        ax.set_xlabel("Wavelength (nm)")

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Compute summary metrics ──
@torch.no_grad()
def compute_metrics(models: dict[str, nn.Module], val_loader, n_wavelengths: int) -> dict:
    metrics = {}
    for name, model in models.items():
        all_pred, all_true = [], []
        for batch in val_loader:
            geo, px, mat, target = (batch["geometry"], batch["params_x"],
                                    batch["material_id"], batch["target"])
            if "cnn" in name.lower():
                pred = model(px, geo[:, -1:], mat)
            else:
                pred = model(geo, mat)
            all_pred.append(pred)
            all_true.append(target)

        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        diff = pred - true

        metrics[name] = {
            "mse": float(torch.mean(diff ** 2)),
            "mae": float(torch.mean(torch.abs(diff))),
            "max_abs_error": float(torch.max(torch.abs(diff))),
            "r2": float(1 - torch.sum(diff ** 2) / torch.sum((true - true.mean()) ** 2)),
            "mse_per_wavelength": torch.mean(diff ** 2, dim=0).tolist(),
        }
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset stats
    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    # Fix relative paths from training script. Always resolve them to PROJECT_ROOT/Data/
    mat_dirs = {k: str(PROJECT_ROOT / "Data" / Path(v).name) for k, v in stats["materials"].items()}
    target_key = stats["target_key"]
    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={list(mat_dirs.keys())}")

    # Rebuild validation set using saved normalization stats
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
    # Subsample if requested
    if args.n_eval < len(val_ds):
        val_ds, _ = random_split(val_ds, [args.n_eval, len(val_ds) - args.n_eval],
                                 generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    print(f"Validation samples: {len(val_ds)}")

    # Load all available models
    all_history = {}
    forward_models = {}

    # ForwardMLP
    mlp_path = ckpt_dir / "forward_mlp.pt"
    if mlp_path.exists():
        model = ForwardMLP(
            n_continuous=n_continuous, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            hidden_dims=(512, 1024, 1024, 1024, 512), activation="snake",
        )
        hist, model = load_checkpoint(mlp_path, model)
        forward_models["forward_mlp"] = model
        all_history["forward_mlp"] = hist
        print("Loaded: forward_mlp")

    # SpatialCNN
    cnn_path = ckpt_dir / "spatial_cnn.pt"
    if cnn_path.exists():
        model = SpatialCNN(
            n_harmonics=n_harmonics, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            n_pixels=256, conv_channels=(64, 128, 256, 512, 256),
            fc_dims=(512, 1024, 512),
        )
        hist, model = load_checkpoint(cnn_path, model)
        forward_models["spatial_cnn"] = model
        all_history["spatial_cnn"] = hist
        print("Loaded: spatial_cnn")

    # SpatialMamba
    mamba_path = ckpt_dir / "spatial_mamba.pt"
    if mamba_path.exists():
        model = SpatialMamba(
            n_harmonics=n_harmonics, n_wavelengths=n_wavelengths,
            n_materials=N_MATERIALS, embed_dim=8,
            n_pixels=256, d_model=512, n_layers=4,
            fc_dims=(512, 1024, 512),
        )
        hist, model = load_checkpoint(mamba_path, model)
        forward_models["spatial_mamba"] = model
        all_history["spatial_mamba"] = hist
        print("Loaded: spatial_mamba")

    # Load histories for inverse models
    for name in ("tandem", "generative_tandem", "cvae"):
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            all_history[name] = ckpt.get("history", {})

    inverse_models = {}
    
    # Tandem
    tandem_path = ckpt_dir / "tandem.pt"
    if tandem_path.exists() and "forward_mlp" in forward_models:
        dec = InverseDecoder(
            n_wavelengths=n_wavelengths, n_geometry=n_continuous,
            n_materials=N_MATERIALS, latent_dim=0,
            hidden_dims=(512, 1024, 1024, 1024, 512),
        )
        tandem = TandemNetwork(inverse_decoder=dec, forward_model=forward_models["forward_mlp"])
        ckpt = torch.load(tandem_path, map_location="cpu", weights_only=False)
        tandem.load_state_dict(ckpt["model_state_dict"])
        tandem.eval()
        inverse_models["tandem"] = tandem
        print("Loaded: tandem")

    # Generative Tandem
    gen_path = ckpt_dir / "generative_tandem.pt"
    if gen_path.exists() and "forward_mlp" in forward_models:
        dec = InverseDecoder(
            n_wavelengths=n_wavelengths, n_geometry=n_continuous,
            n_materials=N_MATERIALS, latent_dim=32,
            hidden_dims=(512, 1024, 1024, 1024, 512),
        )
        gen_tandem = GenerativeTandemNetwork(inverse_decoder=dec, forward_model=forward_models["forward_mlp"], latent_dim=32)
        ckpt = torch.load(gen_path, map_location="cpu", weights_only=False)
        gen_tandem.load_state_dict(ckpt["model_state_dict"])
        gen_tandem.eval()
        inverse_models["generative_tandem"] = gen_tandem
        print("Loaded: generative_tandem")
        
    # Contrastive VAE
    cvae_path = ckpt_dir / "cvae.pt"
    if cvae_path.exists():
        geo_enc = GeometryEncoder(
            n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8,
            latent_dim=64, hidden_dims=(512, 1024, 512),
        )
        geo_dec = GeometryDecoder(
            latent_dim=64, n_geometry=n_continuous,
            n_materials=N_MATERIALS, hidden_dims=(512, 1024, 512),
        )
        spec_enc = SpectrumEncoder(
            n_wavelengths=n_wavelengths, latent_dim=64,
            hidden_dims=(512, 1024, 1024, 512),
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

    # ── Generate plots ──
    print("\nGenerating evaluation plots...")

    if all_history:
        plot_loss_curves(all_history, str(eval_dir / "loss_curves.png"))

    if forward_models:
        plot_forward_parity(forward_models, val_loader,
                           str(eval_dir / "forward_parity.png"), n_wavelengths)
        plot_spectrum_samples(forward_models, val_loader,
                             str(eval_dir / "spectrum_samples.png"), n_wavelengths)

    if inverse_models and "forward_mlp" in forward_models:
        plot_inverse_performance(inverse_models, forward_models["forward_mlp"], val_loader,
                                str(eval_dir / "inverse_performance.png"), n_wavelengths)
        plot_unit_absorptance(inverse_models, forward_models["forward_mlp"],
                             str(eval_dir / "unit_absorptance.png"), n_wavelengths, stats)

    # ── Summary metrics ──
    if forward_models:
        print("\nComputing metrics...")
        metrics = compute_metrics(forward_models, val_loader, n_wavelengths)
        # Remove per-wavelength for the printed summary
        for name, m in metrics.items():
            print(f"\n  {name}:")
            print(f"    MSE: {m['mse']:.6e}")
            print(f"    MAE: {m['mae']:.6f}")
            print(f"    Max Error: {m['max_abs_error']:.6f}")
            print(f"    R²: {m['r2']:.6f}")

        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved: {metrics_path}")

    print(f"\nAll evaluation outputs saved to: {eval_dir}")


if __name__ == "__main__":
    main()
