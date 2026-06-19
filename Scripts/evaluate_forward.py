#!/usr/bin/env python
"""
Evaluate trained forward surrogate models and generate performance report.

Usage:
    python Scripts/evaluate_forward.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4
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
torch.set_float32_matmul_precision("high")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS,
)
from Utils.utils import generate_test_batch
from Utils.checkpoint import load_forward_model, _FORWARD_FILENAMES

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

WAVELENGTHS = np.linspace(300, 1100, 161)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
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
        ax.set_ylabel("Combined Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Forward Training Loss Curves", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_forward_parity(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int):
    n_models = len(models)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, model) in zip(axes, models.items()):
        all_pred, all_true = [], []
        for batch in val_loader:
            geo, px, mat, target = (batch["geometry"], batch["params_x"],
                                    batch["material_id"], batch["target"])
            pred = model(geo, mat)
            all_pred.append(pred)
            all_true.append(target)

        pred = torch.cat(all_pred).mean(dim=1).numpy()
        true = torch.cat(all_true).mean(dim=1).numpy()

        ax.scatter(true, pred, alpha=0.6, s=15, c="blue", edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.8)
        ax.set_xlabel("True Average Absorptance")
        ax.set_ylabel("Pred Average Absorptance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{name.replace('_', ' ').title()}", fontsize=12)

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_spectrum_samples(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int):
    if not models:
        return
    batch = next(iter(val_loader))
    geo, px, mat, target = batch["geometry"], batch["params_x"], batch["material_id"], batch["target"]
    
    selected_indices = []
    mat_counts = {}
    for idx, m_id in enumerate(mat.numpy()):
        m_id = m_id.item()
        if mat_counts.get(m_id, 0) < 2:
            selected_indices.append(idx)
            mat_counts[m_id] = mat_counts.get(m_id, 0) + 1
        if len(selected_indices) >= 6:
            break
            
    # Sort indices so the materials appear in order (e.g. 0, 0, 1, 1, 2, 2)
    selected_indices.sort(key=lambda idx: mat[idx].item())
            
    n_samples = len(selected_indices)
    n_wl_half = n_wavelengths // 2
    mat_names = list(MATERIAL_LIBRARY.keys())
    n_wl_half = n_wavelengths // 2

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 3 * n_samples), squeeze=False, layout="constrained")
        
        for i, idx in enumerate(selected_indices):
            pred = model(geo[idx:idx+1], mat[idx:idx+1])
            mat_idx = mat[idx].item()
            mat_name = mat_names[mat_idx] if mat_idx < len(mat_names) else f"Mat_{mat_idx}"

            for pol_idx, pol_label in enumerate(["p-pol", "s-pol"]):
                ax = axes[i, pol_idx]
                start = pol_idx * n_wl_half
                end = start + n_wl_half
                
                # Plot Truth and Pred
                ax.plot(WAVELENGTHS, target[idx, start:end].numpy(), "k-", lw=2.5, label="Truth", alpha=0.5)
                ax.plot(WAVELENGTHS, pred[0, start:end].numpy(), "--", color=color, lw=2.0, label=name.replace("_", " ").title())
                
                ax.set_xlim(300, 1100)
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(f"Absorptance ({mat_name})")
                if i == 0:
                    ax.set_title(f"Predictions ({pol_label})")
                if i == 0 and pol_idx == 0:
                    ax.legend(fontsize=9)
                if i == n_samples - 1:
                    ax.set_xlabel("Wavelength (nm)")
                    
                # Plot Errors
                ax_err = axes[i, pol_idx + 2]
                error = target[idx, start:end] - pred[0, start:end]
                ax_err.plot(WAVELENGTHS, error.numpy(), "-", color=color, lw=1.5, label=name.replace("_", " ").title())
                
                ax_err.axhline(0, color='k', linestyle='--', lw=1.0)
                ax_err.set_xlim(300, 1100)
                ax_err.set_ylabel(f"Error ({mat_name})")
                if i == 0:
                    ax_err.set_title(f"Error ({pol_label})")
                if i == n_samples - 1:
                    ax_err.set_xlabel("Wavelength (nm)")

        model_save_path = save_path.replace(".png", f"_{name}.png")
        plt.savefig(model_save_path)
        plt.close()
        print(f"  Saved: {model_save_path}")


@torch.no_grad()
def compute_metrics(models: dict[str, nn.Module], val_loader, n_wavelengths: int) -> dict:
    metrics = {}
    for name, model in models.items():
        all_pred, all_true = [], []
        for batch in val_loader:
            geo, px, mat, target = batch["geometry"], batch["params_x"], batch["material_id"], batch["target"]
            pred = model(geo, mat)
            all_pred.append(pred)
            all_true.append(target)

        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        diff = pred - true

        metrics[name] = {
            "mse": float(torch.mean(diff ** 2)),
            "mae": float(torch.mean(torch.abs(diff))),
            "max_abs_error": float(torch.mean(torch.max(torch.abs(diff), dim=1).values)),
            "r2": float(1 - torch.sum(diff ** 2) / torch.sum((true - true.mean()) ** 2)),
        }
    return metrics


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "evaluation" / "forward"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    mat_dirs = {k: str(PROJECT_ROOT / "Data" / Path(v).name) for k, v in stats["materials"].items()}
    target_key = stats["target_key"]
    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={list(mat_dirs.keys())}")

    batch = generate_test_batch(stats)
    val_loader = [batch]

    all_history = {}
    forward_models = {}

    # Build list of all checkpoint paths to evaluate (base + AL)
    ckpt_paths = []
    for fname in _FORWARD_FILENAMES:
        p = ckpt_dir / fname
        if p.exists():
            ckpt_paths.append(p)
    
    al_dir = ckpt_dir / "Active_Learning"
    if al_dir.exists():
        import re
        for p in al_dir.glob("*_al*.pt"):
            if any(p.name.startswith(Path(base_fname).stem) for base_fname in _FORWARD_FILENAMES):
                ckpt_paths.append(p)

    for p in ckpt_paths:
        try:
            model, hist, class_name = load_forward_model(
                p, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_harmonics=n_harmonics
            )
            model.eval()
            
            # Use 'stem' but differentiate AL iterations
            if "_al" in p.name:
                stem = f"{p.stem}"
            else:
                stem = p.stem
                
            forward_models[stem] = model
            all_history[stem] = hist
            print(f"Loaded: {stem} (as {class_name})")
        except Exception as e:
            print(f"Failed to load {p.name}: {e}")



    print("\nGenerating evaluation plots...")
    if all_history:
        plot_loss_curves(all_history, str(eval_dir / "forward_loss_curves.png"))

    if forward_models:
        plot_forward_parity(forward_models, val_loader,
                           str(eval_dir / "forward_parity.png"), n_wavelengths)
        plot_spectrum_samples(forward_models, val_loader,
                             str(eval_dir / "forward_spectrum_samples.png"), n_wavelengths)

        print("\nComputing metrics...")
        metrics = compute_metrics(forward_models, val_loader, n_wavelengths)
        for name, m in metrics.items():
            print(f"\n  {name}:")
            print(f"    MSE: {m['mse']:.6e}")
            print(f"    MAE: {m['mae']:.6f}")
            print(f"    Max Error: {m['max_abs_error']:.6f}")
            print(f"    R²: {m['r2']:.6f}")

        metrics_path = eval_dir / "forward_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved: {metrics_path}")

if __name__ == "__main__":
    main()
