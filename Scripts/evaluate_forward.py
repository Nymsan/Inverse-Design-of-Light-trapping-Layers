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
    MATERIAL_LIBRARY, N_MATERIALS, GratingDataset
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
    p.add_argument("--h_val", nargs="+", type=float, help="Target height in nm, or range (min max)")
    p.add_argument("--h_tolerance", type=float, default=0.5, help="Tolerance for height matching")
    p.add_argument("--inc_val", type=float, default=None, help="Target incident angle in degrees (e.g., 0.0 for normal)")
    p.add_argument("--inc_tolerance", type=float, default=0.5, help="Tolerance for incident angle matching")
    p.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to evaluate, e.g., --bands 500 750 800 900")
    return p.parse_args()

def get_folder_name(args) -> str:
    parts = []
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            parts.append(f"h{int(args.h_val[0])}-{int(args.h_val[1])}")
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            parts.append(f"h{int(h_target)}_tol{args.h_tolerance}")
    if args.inc_val is not None:
        parts.append(f"inc{args.inc_val}_tol{args.inc_tolerance}")
    if args.bands:
        bands_str = "_".join([f"{int(args.bands[i])}-{int(args.bands[i+1])}" for i in range(0, len(args.bands), 2)])
        parts.append(f"bands{bands_str}")
        
    return "_".join(parts) if parts else "all_data"

def format_model_name(name: str) -> str:
    mapping = {
        "forward_mlp": "Forward MLP",
        "spatial_cnn": "Spatial CNN",
        "skip_cnn": "Skip CNN",
        "siren": "SIREN",
        "transformer": "Transformer"
    }
    return mapping.get(name, name.replace("_", " ").title())


def plot_loss_curves(all_history: dict, save_path: str, train_info: str):
    if not all_history:
        return
    n_models = len(all_history)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False, sharey=True, layout="constrained")
    axes = axes[0]

    for ax, (name, hist) in zip(axes, all_history.items()):
        if "train_loss" not in hist:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], label="Train", alpha=0.8)
        if "val_loss" in hist:
            val_epochs = range(1, len(hist["val_loss"]) + 1)
            # If val_loss matches train_loss length exactly, plot them together. Otherwise just plot the available points
            marker = "o" if len(hist["val_loss"]) == 1 else None
            ax.semilogy(val_epochs, hist["val_loss"], label="Val", alpha=0.8, marker=marker)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Combined Loss")
        ax.set_title(format_model_name(name))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(f"Forward Training Loss Curves ({train_info})", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_forward_parity(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int, band_mask: torch.Tensor = None, filter_title: str = ""):
    n_models = len(models)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, model) in zip(axes, models.items()):
        all_pred, all_true, all_mat = [], [], []
        for batch in val_loader:
            geo, px, mat, target = (batch["geometry"], batch["params_x"],
                                    batch["material_id"], batch["target"])
            pred = model(geo, mat)
            if band_mask is not None:
                pred = pred[:, band_mask]
                target = target[:, band_mask]
                
            all_pred.append(pred)
            all_true.append(target)
            all_mat.append(mat)

        pred = torch.cat(all_pred).mean(dim=1).numpy()
        true = torch.cat(all_true).mean(dim=1).numpy()
        mat_all = torch.cat(all_mat).numpy()
        if mat_all.ndim > 1: mat_all = mat_all.argmax(axis=1)

        ax.scatter(true, pred, alpha=0.6, s=15, c="blue", edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.8)
        ax.set_xlabel("True Average Absorptance")
        ax.set_ylabel("Pred Average Absorptance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(format_model_name(name), fontsize=12)
        
        # Create single histogram plot
        save_dir = Path(save_path).parent
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5), layout="constrained")
        
        mat_names = list(MATERIAL_LIBRARY.keys())
        colors = plt.cm.tab10.colors
        
        for m_id in range(len(mat_names)):
            m_mask = (mat_all == m_id)
            if not m_mask.any(): continue
            
            z = m_id + 1
            # Compute fixed bin edges based on min/max of both true and pred
            combined_min = min(true[m_mask].min(), pred[m_mask].min())
            combined_max = max(true[m_mask].max(), pred[m_mask].max())
            bin_width = 0.002
            grid_min = np.floor(combined_min / bin_width) * bin_width
            grid_max = np.ceil(combined_max / bin_width) * bin_width
            num_bins = max(1, int(np.round((grid_max - grid_min) / bin_width)))
            bin_edges = np.linspace(grid_min, grid_max, num_bins + 1)
            
            # True as colored alpha bars (rendered beneath lines)
            ax_hist.hist(true[m_mask], bins=bin_edges, color=colors[m_id], alpha=0.5, label=f"{mat_names[m_id]} (True)", zorder=z)
            # Predicted as colored overlay envelope (rendered on top of all bars)
            ax_hist.plot([], [], color=colors[m_id], linewidth=2.0, label=f"{mat_names[m_id]} (Pred)")
            ax_hist.hist(pred[m_mask], bins=bin_edges, color=colors[m_id], histtype="step", linewidth=2.0, zorder=z)
            
        ax_hist.set_xlabel("Average Absorptance")
        ax_hist.set_ylabel("Count")
        ax_hist.set_yscale("log")
        ax_hist.set_title("Validation Distribution (True vs Predicted)")
        ax_hist.legend()
        
        suptitle = f"{format_model_name(name)} - Performance Histograms"
        if filter_title: suptitle += f"\n({filter_title})"
        fig_hist.suptitle(suptitle, fontsize=12)
        
        hist_save_path = save_dir / f"histogram_{name}.png"
        
        fig_hist.savefig(hist_save_path)
        plt.close(fig_hist)
        print(f"  Saved: {hist_save_path}")

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def plot_spectrum_samples(models: dict[str, nn.Module], val_loader, save_path: str, n_wavelengths: int, n_harmonics: int, global_max_err: float, filter_title: str = ""):
    if not models:
        return
        
    val_set = val_loader.dataset
    import random
    indices = list(range(len(val_set)))
    random.shuffle(indices)
    
    geo_list, mat_list, target_list = [], [], []
    mat_counts = {}
    
    for idx in indices:
        item = val_set[idx]
        m_id = item["material_id"].argmax().item() if item["material_id"].ndim > 0 else item["material_id"].item()
        if mat_counts.get(m_id, 0) < 1:
            geo_list.append(item["geometry"].unsqueeze(0))
            mat_list.append(item["material_id"].unsqueeze(0))
            target_list.append(item["target"].unsqueeze(0))
            mat_counts[m_id] = 1
        if len(geo_list) >= len(MATERIAL_LIBRARY):
            break
            
    combined = sorted(zip(geo_list, mat_list, target_list), key=lambda x: x[1].argmax().item() if x[1].ndim > 0 else x[1].item())
    geo = torch.cat([x[0] for x in combined])
    mat = torch.cat([x[1] for x in combined])
    target = torch.cat([x[2] for x in combined])
    selected_indices = list(range(len(geo)))
            
    n_samples = len(selected_indices)
    n_wl_half = n_wavelengths // 2
    mat_names = list(MATERIAL_LIBRARY.keys())
    n_wl_half = n_wavelengths // 2

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        fig, axes = plt.subplots(n_samples, 5, figsize=(25, 3 * n_samples), squeeze=False, layout="constrained")
        
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
                ax.plot(WAVELENGTHS, pred[0, start:end].numpy(), color=color, ls="--", lw=2, label=format_model_name(name))
                
                ax.set_xlim(300, 1100)
                ax.set_ylim([-0.05, 1.05])
                
                inc_val = geo[idx, -1].item()
                ax.set_ylabel(f"Absorptance ({mat_name})")
                
                # We want the title on every subplot to clearly state the incidence angle
                ax.set_title(f"{pol_label} (inc={inc_val:.1f}°)")
                
                if pol_idx == 0:
                    ax.legend(loc="upper right", fontsize=8)
                if i == n_samples - 1:
                    ax.set_xlabel("Wavelength (nm)")
                    
                # Plot Errors
                ax_err = axes[i, pol_idx + 2]
                error = target[idx, start:end] - pred[0, start:end]
                ax_err.plot(WAVELENGTHS, error.numpy(), "-", color=color, lw=1.5, label=format_model_name(name))
                
                ax_err.axhline(0, color='k', linestyle='--', lw=1.0)
                ax_err.set_xlim(300, 1100)
                ax_err.set_ylim(-global_max_err, global_max_err)
                ax_err.set_ylabel(f"Error ({mat_name})")
                if i == 0:
                    ax_err.set_title(f"Error ({pol_label})")
                if i == n_samples - 1:
                    ax_err.set_xlabel("Wavelength (nm)")

            # Plot Parameters
            ax_geo = axes[i, 4]
            h_val = geo[idx, -2].item()
            inc_val = geo[idx, -1].item()
            
            n_fourier = n_harmonics * 2
            amps = geo[idx, 0:n_fourier:2].numpy()
            phases = geo[idx, 1:n_fourier:2].numpy()
            x_pos = np.arange(1, n_harmonics + 1)
            
            cmap = plt.get_cmap("viridis")
            c_amp = cmap(0.3)
            c_phase = cmap(0.9)
            
            ax_geo.bar(x_pos, amps, color=c_amp, edgecolor="black")
            ax_geo.set_ylabel("Amplitude (nm)", color=c_amp, fontsize=10)
            ax_geo.tick_params(axis='y', labelcolor=c_amp, labelsize=9)
            ax_geo.tick_params(axis='x', labelsize=9)
            
            ax_p2 = ax_geo.twinx()
            ax_p2.plot(x_pos, phases, 'o', color=c_phase, markersize=5, markeredgecolor="black")
            ax_p2.set_ylabel("Phase (rad)", color=c_phase, fontsize=10)
            ax_p2.tick_params(axis='y', labelcolor=c_phase, labelsize=9)
            ax_p2.set_ylim(-0.5, 2 * np.pi + 0.5)
            
            ax_geo.set_title(f"h: {h_val:.0f}nm, inc: {inc_val:.0f}°", fontsize=11)
            if i == n_samples - 1:
                ax_geo.set_xlabel("Harmonic index")

        fig.suptitle(f"Spectrum Samples ({filter_title})", fontsize=16)
        model_save_path = save_path.replace(".png", f"_{name}.png")
        plt.savefig(model_save_path)
        plt.close()
        print(f"  Saved: {model_save_path}")


@torch.no_grad()
def compute_metrics(models: dict[str, nn.Module], val_loader, n_wavelengths: int, band_mask: torch.Tensor = None) -> dict:
    metrics = {}
    for name, model in models.items():
        all_pred, all_true = [], []
        for batch in val_loader:
            geo, px, mat, target = batch["geometry"], batch["params_x"], batch["material_id"], batch["target"]
            pred = model(geo, mat)
            if band_mask is not None:
                pred = pred[:, band_mask]
                target = target[:, band_mask]
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
    eval_dir = ckpt_dir / "evaluation" / "forward" / get_folder_name(args)
    eval_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    prefix = stats.get("dataset_prefixes", ["LHS_Dataset"])[0]
    mat_dirs = {mat: str(PROJECT_ROOT / "Data" / f"{prefix}_{mat}") for mat in stats["materials"]}
    target_key = stats["target_key"]
    print(f"n_continuous={n_continuous}  n_wavelengths={n_wavelengths}  materials={list(mat_dirs.keys())}")

    val_files = {mat: [] for mat in stats["materials"]}
    for mat_name in stats["materials"]:
        d_dir = os.path.join(PROJECT_ROOT, "Data", f"{prefix}_{mat_name}")
        v_file = os.path.join(d_dir, "val_dataset.pt")
        if os.path.exists(v_file):
            val_files[mat_name].append(v_file)
        else:
            raise FileNotFoundError(f"Missing {v_file}")
            
    val_set = GratingDataset(
        val_files, target_key=stats["target_key"],
        geo_min=stats["geo_min"], geo_max=stats["geo_max"]
    )
    
    # Filter validation set
    valid_mask = torch.ones(len(val_set), dtype=torch.bool)
    if args.h_val is not None:
        h = val_set.geometry[:, -2]
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            valid_mask &= (h >= args.h_val[0]) & (h <= args.h_val[1])
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            valid_mask &= (torch.abs(h - h_target) <= args.h_tolerance)
            
    if args.inc_val is not None:
        inc = val_set.geometry[:, -1]
        valid_mask &= (torch.abs(inc - args.inc_val) <= args.inc_tolerance)
        
    num_valid = valid_mask.sum().item()
    if num_valid == 0:
        print("No samples match the given filters!")
        return
        
    if num_valid < len(val_set):
        val_set.geometry = val_set.geometry[valid_mask]
        val_set.params_x = val_set.params_x[valid_mask]
        val_set.material_id = val_set.material_id[valid_mask]
        val_set.target = val_set.target[valid_mask]
        print(f"Filtered validation set from {len(valid_mask)} down to {num_valid} samples.")
        
    band_mask = None
    if args.bands:
        wl_mask = np.zeros(len(WAVELENGTHS), dtype=bool)
        for i in range(0, len(args.bands), 2):
            wl_mask |= (WAVELENGTHS >= args.bands[i]) & (WAVELENGTHS <= args.bands[i+1])
        band_mask = torch.from_numpy(np.concatenate([wl_mask, wl_mask]))
        
    print(f"Loaded real validation set with {len(val_set)} samples.")
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

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



    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)

    import re
    train_info = "Full Trainset"
    m = re.search(r"frac_([0-9.]+)", ckpt_dir.name)
    if m:
        frac = float(m.group(1))
        train_info = f"Fraction: {frac}"
        
    plot_loss_curves(all_history, str(eval_dir / "forward_loss_curves.png"), train_info)

    title_parts = []
    if train_info:
        title_parts.append(train_info)
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            title_parts.append(f"h={args.h_val[0]}-{args.h_val[1]}nm")
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            title_parts.append(f"h={h_target}±{args.h_tolerance}nm")
    if args.inc_val is not None:
        title_parts.append(f"inc={args.inc_val}°±{args.inc_tolerance}°")
    if args.bands:
        title_parts.append("Matched Bands")
    filter_title = " | ".join(title_parts) if title_parts else "All Geometries, All Angles"

    if forward_models:
        plot_forward_parity(forward_models, val_loader,
                           str(eval_dir / "forward_parity.png"), n_wavelengths, band_mask, filter_title)
        
        print("\nComputing metrics...")
        metrics = compute_metrics(forward_models, val_loader, n_wavelengths, band_mask)
        
        global_max_err = 0.0
        for name, m in metrics.items():
            print(f"\n  {format_model_name(name)}:")
            print(f"    MSE: {m['mse']:.6e}")
            print(f"    MAE: {m['mae']:.6f}")
            print(f"    Max Error: {m['max_abs_error']:.6f}")
            print(f"    R²: {m['r2']:.6f}")
            global_max_err = max(global_max_err, m['max_abs_error'])
            
        global_max_err = global_max_err * 1.05 # 5% padding
        if global_max_err < 0.05: global_max_err = 0.05 # minimum limit

        metrics_path = eval_dir / "forward_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved: {metrics_path}")

        plot_spectrum_samples(
            forward_models, val_loader, str(eval_dir / "forward_spectrum_samples.png"), n_wavelengths, n_harmonics, global_max_err, filter_title
        )

if __name__ == "__main__":
    main()
