#!/usr/bin/env python
"""
Evaluate a trained SkipCNN3D checkpoint and generate a performance report.

Usage:
    python Scripts/evaluate_3d_forward.py --ckpt Checkpoints/3D_Si_TiO2_Si3N4/skipcnn3d.pt

Outputs (saved alongside the checkpoint in an 'evaluation/' sub-directory):
    loss_curves.png        — train / val loss over epochs
    parity_p.png           — scatter: true vs pred  (p-pol)
    parity_s.png           — scatter: true vs pred  (s-pol)
    histogram_p.png        — distribution true vs pred absorptance (p-pol)
    histogram_s.png        — distribution true vs pred absorptance (s-pol)
    samples.png            — per-sample error bar charts
    metrics.json           — MSE, MAE, max error, R²
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models_3d import SkipCNN3D, Grating3DDataset

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

POL_LABELS = ["p-pol", "s-pol"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_state_dict(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_checkpoint(ckpt_path: Path) -> tuple[SkipCNN3D, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["model_config"]
    model = SkipCNN3D(**cfg)
    model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]), strict=True)
    model.eval()
    return model, ckpt.get("history", {})


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred, true) each (N, 2)."""
    preds, trues = [], []
    for batch in loader:
        px  = batch["params_x"].to(device)
        py  = batch["params_y"].to(device)
        h   = batch["h"].to(device)
        wl  = batch["wavelength"].to(device)
        tgt = batch["target"]
        out = model(px, py, h, wl).cpu()
        preds.append(out)
        trues.append(tgt)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curves(history: dict, save_path: Path):
    if not history.get("train_loss"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    for ax, key, label in zip(
        axes,
        [("train_loss", "val_loss"), ("val_mae", "val_max_err")],
        [("Train loss", "Val loss"), ("Val MAE", "Val max error")],
    ):
        for k, lbl in zip(key, label):
            if k in history and history[k]:
                ax.semilogy(range(1, len(history[k]) + 1), history[k], label=lbl)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Training & Validation Loss")
    axes[1].set_title("Validation Error Metrics")
    fig.suptitle("SkipCNN3D — Training Curves", fontsize=14)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_parity(pred: np.ndarray, true: np.ndarray, pol_idx: int, save_path: Path):
    """Parity scatter for one polarisation."""
    p = pred[:, pol_idx]
    t = true[:, pol_idx]
    err = np.abs(p - t)

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    sc = ax.scatter(t, p, c=err, cmap="plasma", alpha=0.5, s=8,
                    vmin=0, vmax=err.max(), edgecolors="none")
    plt.colorbar(sc, ax=ax, label="|Error|")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"True absorptance ({POL_LABELS[pol_idx]})")
    ax.set_ylabel(f"Predicted absorptance ({POL_LABELS[pol_idx]})")
    ax.set_title(f"Parity plot — {POL_LABELS[pol_idx]}")

    # Annotate R² and MAE
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2  = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(err)
    ax.text(0.04, 0.94, f"R² = {r2:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_histogram(pred: np.ndarray, true: np.ndarray, pol_idx: int, save_path: Path):
    """Overlapping histograms of true vs predicted absorptance."""
    p = pred[:, pol_idx]
    t = true[:, pol_idx]
    combined_min = min(t.min(), p.min())
    combined_max = max(t.max(), p.max())
    bins = np.linspace(combined_min, combined_max, 60)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.hist(t, bins=bins, alpha=0.55, color="#2196F3", label="True")
    ax.hist(p, bins=bins, histtype="step", lw=2.0, color="#F44336", label="Predicted")
    ax.set_xlabel(f"Absorptance ({POL_LABELS[pol_idx]})")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"Distribution — {POL_LABELS[pol_idx]}")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_error_distribution(pred: np.ndarray, true: np.ndarray, save_path: Path):
    """Signed error histograms for both polarisations side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = ["#2196F3", "#E91E63"]

    for ax, pol_idx, color in zip(axes, [0, 1], colors):
        err = pred[:, pol_idx] - true[:, pol_idx]
        ax.hist(err, bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(np.mean(err), color="red", lw=1.5, ls=":", label=f"mean={np.mean(err):.4f}")
        ax.set_xlabel(f"Pred − True  ({POL_LABELS[pol_idx]})")
        ax.set_ylabel("Count")
        ax.set_title(f"Signed Error — {POL_LABELS[pol_idx]}")
        ax.legend(fontsize=10)
        ax.set_yscale("log")

    fig.suptitle("SkipCNN3D — Error Distribution", fontsize=13)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_samples(model, val_set, device, save_path: Path, n_samples: int = 8, seed: int = 0):
    """
    Bar-chart style sample plots showing true vs predicted [p, s] absorptance
    for random validation samples, plus grating geometry (amplitudes/phases).
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(val_set), size=min(n_samples, len(val_set)), replace=False)
    indices = sorted(indices)

    fig, axes = plt.subplots(len(indices), 3, figsize=(14, 3 * len(indices)),
                              squeeze=False, layout="constrained")

    c_true = "#2196F3"
    c_pred = "#F44336"
    harm_x = np.arange(1, 6)   # harmonic indices 1–5

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            item  = val_set[idx]
            px    = item["params_x"].unsqueeze(0).to(device)
            py    = item["params_y"].unsqueeze(0).to(device)
            h     = item["h"].unsqueeze(0).to(device)
            wl    = item["wavelength"].unsqueeze(0).to(device)
            pred  = model(px, py, h, wl)[0].cpu().numpy()
            true  = item["target"].numpy()
            h_val = item["h"].item()
            wl_val= item["wavelength"].item()

            # ── left: p/s absorptance bar chart ──────────────────────────────
            ax0 = axes[row, 0]
            x   = np.array([0, 1])
            w   = 0.35
            ax0.bar(x - w/2, true, w, label="True",      color=c_true, alpha=0.8)
            ax0.bar(x + w/2, pred, w, label="Predicted", color=c_pred, alpha=0.8)
            ax0.set_xticks(x); ax0.set_xticklabels(["p-pol", "s-pol"])
            ax0.set_ylim(0, 1.1)
            ax0.set_ylabel("Absorptance")
            ax0.set_title(f"Sample {idx}  |  h={h_val:.0f} nm, λ={wl_val:.0f} nm")
            if row == 0: ax0.legend(fontsize=9)

            # Annotate absolute errors
            for xi, (t, p) in enumerate(zip(true, pred)):
                ax0.text(xi, max(t, p) + 0.03, f"Δ={abs(p-t):.3f}", ha="center",
                         fontsize=8, color="grey")

            # ── middle: x-harmonic amps & phases ────────────────────────────
            ax1 = axes[row, 1]
            amps_x   = item["params_x"][:, 0].numpy()
            phases_x = item["params_x"][:, 1].numpy()
            cmap = plt.get_cmap("viridis")
            ax1.bar(harm_x, amps_x, color=cmap(0.3), edgecolor="k", label="Amp X")
            ax1.set_ylabel("Amplitude (nm)", color=cmap(0.3))
            ax1.tick_params(axis="y", labelcolor=cmap(0.3))
            ax1b = ax1.twinx()
            ax1b.plot(harm_x, phases_x, "o", color=cmap(0.85), ms=6,
                      markeredgecolor="k", label="Phase X")
            ax1b.set_ylim(-0.3, 2 * np.pi + 0.3)
            ax1b.set_ylabel("Phase (rad)", color=cmap(0.85))
            ax1b.tick_params(axis="y", labelcolor=cmap(0.85))
            ax1.set_title("X harmonics")
            ax1.set_xlabel("Harmonic index")

            # ── right: y-harmonic amps & phases ─────────────────────────────
            ax2 = axes[row, 2]
            amps_y   = item["params_y"][:, 0].numpy()
            phases_y = item["params_y"][:, 1].numpy()
            ax2.bar(harm_x, amps_y, color=cmap(0.3), edgecolor="k", label="Amp Y")
            ax2.set_ylabel("Amplitude (nm)", color=cmap(0.3))
            ax2.tick_params(axis="y", labelcolor=cmap(0.3))
            ax2b = ax2.twinx()
            ax2b.plot(harm_x, phases_y, "s", color=cmap(0.85), ms=6,
                      markeredgecolor="k", label="Phase Y")
            ax2b.set_ylim(-0.3, 2 * np.pi + 0.3)
            ax2b.set_ylabel("Phase (rad)", color=cmap(0.85))
            ax2b.tick_params(axis="y", labelcolor=cmap(0.85))
            ax2.set_title("Y harmonics")
            ax2.set_xlabel("Harmonic index")

    fig.suptitle("SkipCNN3D — Validation Samples", fontsize=14)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_wavelength_dependence(pred: np.ndarray, true: np.ndarray,
                               wavelengths: np.ndarray, save_path: Path,
                               n_bins: int = 40):
    """
    Bin samples by wavelength and plot mean MAE vs wavelength for p and s pol.
    Shows where the model struggles across the spectrum.
    """
    wl_min, wl_max = wavelengths.min(), wavelengths.max()
    edges = np.linspace(wl_min, wl_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    mae_p, mae_s, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (wavelengths >= lo) & (wavelengths < hi)
        if mask.sum() > 0:
            mae_p.append(np.mean(np.abs(pred[mask, 0] - true[mask, 0])))
            mae_s.append(np.mean(np.abs(pred[mask, 1] - true[mask, 1])))
            counts.append(mask.sum())
        else:
            mae_p.append(np.nan); mae_s.append(np.nan); counts.append(0)

    mae_p = np.array(mae_p); mae_s = np.array(mae_s); counts = np.array(counts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    ax1.plot(centres, mae_p, "o-", color="#2196F3", ms=4, label="p-pol MAE")
    ax1.plot(centres, mae_s, "s-", color="#E91E63", ms=4, label="s-pol MAE")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("MAE vs Wavelength")

    ax2.bar(centres, counts, width=(wl_max - wl_min) / n_bins * 0.8,
            color="grey", alpha=0.7, label="Sample count")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("# Samples in bin")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SkipCNN3D — Error vs Wavelength", fontsize=13)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SkipCNN3D checkpoint")
    p.add_argument("--ckpt",      required=True, help="Path to skipcnn3d.pt checkpoint")
    p.add_argument("--data_dir",  type=str, default="Data",
                   help="Root data directory (default: Data/)")
    p.add_argument("--materials", nargs="+", default=None,
                   help="Override materials list (default: read from checkpoint dir's dataset_stats.pt)")
    p.add_argument("--target_key", type=str, default=None,
                   help="Override target key (default: read from dataset_stats.pt)")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--n_samples",  type=int, default=8,
                   help="Number of random sample plots")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    ckpt_dir  = ckpt_path.parent
    eval_dir  = ckpt_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset stats (materials, target key) ────────────────────────
    stats_path = ckpt_dir / "dataset_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    else:
        stats = {}

    materials   = args.materials  or stats.get("materials",  ["Si", "TiO2", "Si3N4"])
    target_key  = args.target_key or stats.get("target_key", "A_film_normal")
    h_min  = stats.get("h_min")
    h_max  = stats.get("h_max")
    wl_min = stats.get("wl_min")
    wl_max = stats.get("wl_max")

    print(f"Materials  : {materials}")
    print(f"Target key : {target_key}")

    # ── Load val dataset ─────────────────────────────────────────────────
    data_root = PROJECT_ROOT / args.data_dir
    val_files = []
    for mat in materials:
        v = data_root / f"LHS_3D_Dataset_{mat}" / "val_dataset.pt"
        if not v.exists():
            raise FileNotFoundError(f"Missing {v}")
        val_files.append(str(v))

    val_set = Grating3DDataset(
        val_files, target_key=target_key,
        h_min=h_min, h_max=h_max, wl_min=wl_min, wl_max=wl_max,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Val samples: {len(val_set)}")

    # ── Load model ────────────────────────────────────────────────────────
    model, history = load_checkpoint(ckpt_path)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    # ── Inference ─────────────────────────────────────────────────────────
    print("\nRunning inference on validation set...")
    pred, true = run_inference(model, val_loader, device)
    wavelengths = val_set.wavelength.numpy()

    # ── Metrics ───────────────────────────────────────────────────────────
    def compute_metrics(p: np.ndarray, t: np.ndarray, label: str) -> dict:
        """
        p, t : (N, 2)  — predicted and true absorptances [p-pol, s-pol]

        R² note
        -------
        When computed over *mixed wavelengths* (300–1100 nm), SS_tot is large
        because absorptance varies strongly with wavelength.  A model that merely
        predicts the wavelength-averaged mean already scores high, so R² is
        misleadingly optimistic.

        When computed over the *495 nm subset only*, every sample shares the same
        wavelength, so SS_tot reflects only geometry-driven variation.  That R²
        is a genuine measure of whether the model captures shape effects.
        """
        out = {"_note": label}
        for pol_idx, pol in enumerate(["p", "s"]):
            diff   = p[:, pol_idx] - t[:, pol_idx]
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((t[:, pol_idx] - t[:, pol_idx].mean()) ** 2)
            out[pol] = {
                "n_samples":      int(len(p)),
                "mse":            float(np.mean(diff ** 2)),
                "mae":            float(np.mean(np.abs(diff))),
                "max_abs_error":  float(np.max(np.abs(diff))),
                "mean_abs_error": float(np.mean(np.max(np.abs(p - t), axis=1))),
                "r2":             float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            }
        return out

    def print_metrics(m: dict):
        print(f"\n  [{m['_note']}]  n={m['p']['n_samples']}")
        for pol in ["p", "s"]:
            mi = m[pol]
            print(f"  {POL_LABELS[pol == 's']}:")
            print(f"    MAE           : {mi['mae']:.6f}")
            print(f"    MSE           : {mi['mse']:.6e}")
            print(f"    Max abs error : {mi['max_abs_error']:.6f}")
            print(f"    R²            : {mi['r2']:.6f}")

    # All wavelengths
    metrics_all = compute_metrics(pred, true, "All wavelengths — R² inflated by spectral variance")

    # 495 nm only
    ref_mask = np.isclose(wavelengths, 495.0, atol=0.5)
    if ref_mask.sum() > 0:
        metrics_495 = compute_metrics(pred[ref_mask], true[ref_mask],
                                      "495 nm only — R² reflects geometry variation")
    else:
        metrics_495 = None
        print("  (no 495 nm samples found in val set)")

    print("\n" + "=" * 55)
    print("Validation Metrics")
    print("=" * 55)
    print_metrics(metrics_all)
    if metrics_495:
        print()
        print_metrics(metrics_495)

    all_metrics = {"all_wavelengths": metrics_all}
    if metrics_495:
        all_metrics["ref_495nm"] = metrics_495

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: {eval_dir / 'metrics.json'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_loss_curves(history, eval_dir / "loss_curves.png")

    # --- All wavelengths ---
    for pol_idx in [0, 1]:
        pol = ["p", "s"][pol_idx]
        plot_parity(pred, true, pol_idx, eval_dir / f"parity_{pol}.png")
        plot_histogram(pred, true, pol_idx, eval_dir / f"histogram_{pol}.png")

    plot_error_distribution(pred, true, eval_dir / "error_distribution.png")
    plot_wavelength_dependence(pred, true, wavelengths,
                               eval_dir / "error_vs_wavelength.png")

    # --- 495 nm subset (fixed-wavelength: R² is meaningful here) ---
    if ref_mask.sum() > 0:
        pred_495 = pred[ref_mask]
        true_495 = true[ref_mask]
        print(f"\nGenerating 495 nm subset plots ({ref_mask.sum()} samples)...")
        for pol_idx in [0, 1]:
            pol = ["p", "s"][pol_idx]
            plot_parity(pred_495, true_495, pol_idx,
                        eval_dir / f"parity_{pol}_495nm.png")
            plot_histogram(pred_495, true_495, pol_idx,
                           eval_dir / f"histogram_{pol}_495nm.png")
        plot_error_distribution(pred_495, true_495,
                                eval_dir / "error_distribution_495nm.png")

    plot_samples(model, val_set, device, eval_dir / "samples.png",
                 n_samples=args.n_samples)

    print(f"\nAll outputs saved to: {eval_dir}")


if __name__ == "__main__":
    main()
