#!/usr/bin/env python
"""
Sanity script: find top-k dataset structures by average absorptance (or Jsc),
then re-validate them with a finer Torcwa wavelength grid and compare.

No model required.

Usage examples
--------------
# Basic – top 3 per material, 400-point validation grid:
    python Scripts/validate_dataset_top.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4 \
        --top_k 3 --eval_resolution 400

# Jsc-optimised, band-limited, fixed height:
    python Scripts/validate_dataset_top.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4 \
        --top_k 2 --eval_resolution 300 --optimize_jsc \
        --bands 500 1100 --h_val 1000
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.evaluate_dataset_baseline import get_dataset_baseline
from Utils.utils import RCWAConfig, get_absorptance_curve, sun_weights, get_jsc_scaling_factor
from Utils.models import MATERIAL_LIBRARY

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Re-validate top dataset structures with a finer Torcwa grid."
    )
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    p.add_argument("--top_k", type=int, default=1,
                   help="Number of top structures to validate per material")
    p.add_argument("--eval_resolution", type=int, default=None,
                   help="Number of wavelengths in the fine Torcwa grid (default: 2× dataset resolution)")
    p.add_argument("--bands", nargs="+", type=float,
                   help="Wavelength band pairs to optimise, e.g. --bands 500 750 800 900")
    p.add_argument("--h_val", nargs="+", type=float,
                   help="Fix/range height in nm, e.g. --h_val 1000 or --h_val 500 1500")
    p.add_argument("--h_tolerance", type=float, default=0.5,
                   help="Tolerance (nm) for height matching (only used when h_val is a single value)")
    p.add_argument("--inc_val", type=float, default=None,
                   help="Fix incident angle in degrees (None = all angles)")
    p.add_argument("--inc_tolerance", type=float, default=0.5,
                   help="Tolerance (deg) for inc_val matching")
    p.add_argument("--optimize_jsc", action="store_true",
                   help="Rank structures by Jsc instead of mean absorptance")
    p.add_argument("--order_N", type=int, default=None,
                   help="RCWA order N override for Torcwa validation")
    p.add_argument("--height_per_layer", type=float, default=None,
                   help="Height-per-layer override for Torcwa validation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_avg_abs(curve_p: np.ndarray, curve_s: np.ndarray, mask: np.ndarray) -> float:
    """Mean absorptance over masked wavelengths, averaged across polarisations."""
    return float((curve_p[mask].mean() + curve_s[mask].mean()) / 2.0)


def compute_jsc(curve_p: np.ndarray, curve_s: np.ndarray,
                wavelengths: np.ndarray, mask: np.ndarray,
                inc_ang_deg: float) -> float:
    """Pseudo Jsc [mA/cm²] averaged across polarisations."""
    wls_t = torch.tensor(wavelengths, dtype=torch.float32)
    flux = sun_weights(wls_t).numpy() * wavelengths        # S(λ) * λ
    factor = get_jsc_scaling_factor(len(wavelengths))
    cos_theta = float(np.cos(inc_ang_deg * np.pi / 180.0))

    jsc_p = float((curve_p * mask * flux).sum() * factor * cos_theta)
    jsc_s = float((curve_s * mask * flux).sum() * factor * cos_theta)
    return (jsc_p + jsc_s) / 2.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    # ---- parse bands ----
    bands = []
    if args.bands:
        if len(args.bands) % 2 != 0:
            print("Error: --bands must be pairs of min max values")
            return
        for i in range(0, len(args.bands), 2):
            bands.append((args.bands[i], args.bands[i + 1]))

    # ---- load dataset results ----
    print("Scanning dataset for top structures …")
    results, stats = get_dataset_baseline(
        ckpt_dir, bands=bands,
        h_val=args.h_val, h_tolerance=args.h_tolerance,
        inc_val=args.inc_val, inc_tolerance=args.inc_tolerance,
        optimize_jsc=args.optimize_jsc
    )

    n_harmonics = stats["n_harmonics"]
    n_wl_dataset = stats["n_wavelengths"] // 2          # per-polarisation
    DATASET_WLS = np.linspace(300, 1100, n_wl_dataset)

    eval_n_wl = args.eval_resolution if args.eval_resolution is not None else n_wl_dataset * 2
    FINE_WLS = np.linspace(300, 1100, eval_n_wl)
    print(f"Dataset grid : {n_wl_dataset} wavelengths/pol")
    print(f"Fine Torcwa  : {eval_n_wl} wavelengths/pol")

    # ---- band mask (applied to whichever grid) ----
    def make_mask(wls: np.ndarray) -> np.ndarray:
        if not bands:
            return np.ones(len(wls), dtype=bool)
        m = np.zeros(len(wls), dtype=bool)
        for bmin, bmax in bands:
            m |= (wls >= bmin) & (wls <= bmax)
        return m

    dataset_mask = make_mask(DATASET_WLS)
    fine_mask    = make_mask(FINE_WLS)

    # ---- RCWA base config (read from first batch file if available) ----
    trained_mat_names = stats["materials"]
    first_batch = (PROJECT_ROOT / "Data"
                   / f"LHS_Dataset_{trained_mat_names[0]}" / "batch_0000.pt")
    if first_batch.exists():
        rcwa_config_dict = (
            torch.load(first_batch, map_location="cpu", weights_only=False)
            .get("metadata", {}).get("config", {})
        )
    else:
        rcwa_config_dict = {}

    # ---- output directory ----
    tag_parts = []
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            tag_parts.append(f"h{int(args.h_val[0])}-{int(args.h_val[1])}")
        else:
            h_t = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            tag_parts.append(f"h{int(h_t)}")
    if args.inc_val is not None:
        tag_parts.append(f"inc{args.inc_val}")
    if bands:
        tag_parts.append("_".join(f"{int(b[0])}-{int(b[1])}" for b in bands))
    if args.optimize_jsc:
        tag_parts.append("jsc")
    folder = "_".join(tag_parts) if tag_parts else "all_data"

    out_dir = ckpt_dir / "evaluation" / "dataset_fine_validation" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-material loop ----
    all_metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mat_name, res in results.items():
        metric = res["metric"]
        valid_idx = torch.where(metric >= 0)[0]
        if len(valid_idx) == 0:
            print(f"[{mat_name}] No valid structures – skipping.")
            continue

        k = min(args.top_k, len(valid_idx))
        top_local = torch.topk(metric[valid_idx], k, largest=True).indices
        top_idx   = valid_idx[top_local]

        targets = res["targets"]   # shape [N, 2*n_wl_dataset]
        geos    = res["geos"]      # shape [N, n_harmonics*2+2]

        print(f"\n[{mat_name}] Running Torcwa on top-{k} structure(s) …")

        # --- figure: k rows × 3 cols (p-pol, s-pol, structure) ---
        fig, axes = plt.subplots(k, 3, figsize=(21, 6 * k), squeeze=False)

        cmap = plt.cm.viridis
        c_dataset = cmap(0.25)
        c_fine    = cmap(0.75)

        for row, idx in enumerate(tqdm(top_idx, desc=f"  {mat_name}")):
            geo = geos[idx]
            n_harm = (len(geo) - 2) // 2
            px = geo[:n_harm * 2].view(-1, 2).cpu()   # [n_harm, 2]
            h_val_nm  = geo[-2].item()
            inc_ang_deg = geo[-1].item()

            # ---- dataset curve (already stored) ----
            ds_p = targets[idx, :n_wl_dataset].numpy()
            ds_s = targets[idx, n_wl_dataset:].numpy()

            ds_avg_abs = compute_avg_abs(ds_p, ds_s, dataset_mask)
            ds_jsc     = compute_jsc(ds_p, ds_s, DATASET_WLS, dataset_mask, inc_ang_deg)

            # ---- RCWA config ----
            cfg = RCWAConfig(**rcwa_config_dict)
            cfg.h       = h_val_nm
            cfg.inc_ang = (inc_ang_deg + 1e-3) * np.pi / 180.0
            cfg.azi_ang = 1e-3 * np.pi / 180.0

            if mat_name.endswith("_Ag"):
                cfg.grating_material = mat_name[:-3]
                cfg.reflector_type   = "Ag"
            else:
                cfg.grating_material = mat_name
                cfg.reflector_type   = "pec"

            if args.order_N is not None:
                cfg.order_N = args.order_N
            if args.height_per_layer is not None:
                cfg.height_per_layer = args.height_per_layer

            # ---- Torcwa fine grid ----
            A_film, _ = get_absorptance_curve(
                params_x=px,
                params_y=None,
                wavelengths=torch.from_numpy(FINE_WLS).double(),
                config=cfg,
                show_progress=True,
            )
            fine_p = A_film[:, 0].cpu().numpy()
            fine_s = A_film[:, 1].cpu().numpy()

            fine_avg_abs = compute_avg_abs(fine_p, fine_s, fine_mask)
            fine_jsc     = compute_jsc(fine_p, fine_s, FINE_WLS, fine_mask, inc_ang_deg)

            rank = row + 1
            metric_label  = "Avg Abs" if not args.optimize_jsc else "Jsc [mA/cm²]"
            ds_metric_val = ds_avg_abs    if not args.optimize_jsc else ds_jsc
            fine_metric_val = fine_avg_abs if not args.optimize_jsc else fine_jsc

            delta = fine_metric_val - ds_metric_val

            print(
                f"  Rank {rank}: dataset {metric_label}={ds_metric_val:.4f} | "
                f"fine {metric_label}={fine_metric_val:.4f} | "
                f"Δ={delta:+.4f}"
            )

            # always report both numbers
            print(
                f"           dataset AvgAbs={ds_avg_abs:.4f} / Jsc={ds_jsc:.4f}  |  "
                f"fine    AvgAbs={fine_avg_abs:.4f} / Jsc={fine_jsc:.4f}"
            )

            all_metrics.append({
                "material":          mat_name,
                "rank":              rank,
                "h_nm":              h_val_nm,
                "inc_ang_deg":       inc_ang_deg,
                "dataset_n_wl":      n_wl_dataset,
                "fine_n_wl":         eval_n_wl,
                "dataset_avg_abs":   ds_avg_abs,
                "dataset_jsc":       ds_jsc,
                "fine_avg_abs":      fine_avg_abs,
                "fine_jsc":          fine_jsc,
                "delta_avg_abs":     fine_avg_abs - ds_avg_abs,
                "delta_jsc":         fine_jsc - ds_jsc,
            })

            # ---- plotting ----
            def _plot_pol(ax, wls_ds, curve_ds, wls_fine, curve_fine, pol_label):
                if bands:
                    for bmin, bmax in bands:
                        ax.axvspan(bmin, bmax, color="gray", alpha=0.12)
                ax.plot(wls_ds,   curve_ds,   lw=2.5, color=c_dataset,
                        label=f"Dataset ({n_wl_dataset} pts): AvgAbs={np.mean(curve_ds[dataset_mask]):.3f}")
                ax.plot(wls_fine, curve_fine, lw=2.0, color=c_fine, alpha=0.85,
                        label=f"Torcwa fine ({eval_n_wl} pts): AvgAbs={np.mean(curve_fine[fine_mask]):.3f}")
                ax.set_xlim(300, 1100)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Absorptance")
                ax.set_title(
                    f"Rank {rank}: {mat_name} ({pol_label})\n"
                    f"h={h_val_nm:.0f} nm, θ={inc_ang_deg:.1f}°  |  "
                    f"Dataset {metric_label}={ds_metric_val:.4f} → Fine={fine_metric_val:.4f}  Δ={delta:+.4f}",
                    fontsize=12
                )
                if row == 0:
                    ax.legend(fontsize=10)

            _plot_pol(axes[row, 0], DATASET_WLS, ds_p, FINE_WLS, fine_p, "P-pol")
            _plot_pol(axes[row, 1], DATASET_WLS, ds_s, FINE_WLS, fine_s, "S-pol")

            # structure cross-section
            ax_xs = axes[row, 2]
            n_harm_r = (len(geo) - 2) // 2
            amps   = geo[0:2*n_harm_r:2].numpy()
            phases = geo[1:2*n_harm_r:2].numpy()
            r_grid = np.linspace(0, rcwa_config_dict.get("grating_period", 1000), 256)
            h_idx  = np.arange(1, n_harm_r + 1)
            arg    = 2.0 * np.pi * h_idx[:, None] * r_grid[None, :] / rcwa_config_dict.get("grating_period", 1000) - phases[:, None]
            prof   = amps[:, None] * np.cos(arg)
            grating_h = 2.0 * amps.sum()
            profile   = grating_h / 2.0 + prof.sum(axis=0)

            ax_xs.plot(r_grid, profile, "k-", lw=2)
            ax_xs.fill_between(r_grid, 0, profile, color=cmap(0.6), alpha=0.4)
            ax_xs.set_xlim(0, rcwa_config_dict.get("grating_period", 1000))
            ax_xs.set_ylim(0, max(profile.max() * 1.25, 10))
            ax_xs.set_xlabel("x (nm)")
            ax_xs.set_ylabel("Height (nm)")
            ax_xs.set_title(f"Grating cross-section\nh={h_val_nm:.0f} nm, θ={inc_ang_deg:.1f}°", fontsize=12)

        fig.suptitle(
            f"Dataset top-{k} → Fine Torcwa Validation  |  {mat_name}  "
            f"({'Jsc' if args.optimize_jsc else 'AvgAbs'} ranked, "
            f"{n_wl_dataset}→{eval_n_wl} pts)",
            fontsize=15, y=1.01
        )
        fig.tight_layout()
        save_path = out_dir / f"fine_validation_{mat_name}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")

    # ---- summary bar chart ----
    if all_metrics:
        mat_labels  = [f"{m['material']} R{m['rank']}" for m in all_metrics]
        ds_abs_vals = [m["dataset_avg_abs"] for m in all_metrics]
        fi_abs_vals = [m["fine_avg_abs"]    for m in all_metrics]
        ds_jsc_vals = [m["dataset_jsc"]     for m in all_metrics]
        fi_jsc_vals = [m["fine_jsc"]        for m in all_metrics]

        x = np.arange(len(mat_labels))
        w = 0.35

        fig2, (ax_a, ax_j) = plt.subplots(1, 2, figsize=(max(10, len(mat_labels) * 2.5), 6))

        # avg abs comparison
        ax_a.bar(x - w/2, ds_abs_vals, w, label=f"Dataset ({n_wl_dataset} pts)", color=c_dataset)
        ax_a.bar(x + w/2, fi_abs_vals, w, label=f"Fine Torcwa ({eval_n_wl} pts)", color=c_fine)
        ax_a.set_xticks(x); ax_a.set_xticklabels(mat_labels, rotation=25, ha="right")
        ax_a.set_ylabel("Mean Absorptance"); ax_a.set_title("Average Absorptance Comparison")
        ax_a.legend(); ax_a.set_ylim(0, 1.05)
        for i, (d, f_) in enumerate(zip(ds_abs_vals, fi_abs_vals)):
            ax_a.annotate(f"{f_-d:+.3f}", xy=(x[i] + w/2, f_), ha="center", va="bottom", fontsize=9)

        # jsc comparison
        ax_j.bar(x - w/2, ds_jsc_vals, w, label=f"Dataset ({n_wl_dataset} pts)", color=c_dataset)
        ax_j.bar(x + w/2, fi_jsc_vals, w, label=f"Fine Torcwa ({eval_n_wl} pts)", color=c_fine)
        ax_j.set_xticks(x); ax_j.set_xticklabels(mat_labels, rotation=25, ha="right")
        ax_j.set_ylabel("Pseudo Jsc [mA/cm²]"); ax_j.set_title("Jsc Comparison")
        ax_j.legend()
        for i, (d, f_) in enumerate(zip(ds_jsc_vals, fi_jsc_vals)):
            ax_j.annotate(f"{f_-d:+.3f}", xy=(x[i] + w/2, f_), ha="center", va="bottom", fontsize=9)

        fig2.suptitle(
            f"Dataset ({n_wl_dataset} pts) vs Fine Torcwa ({eval_n_wl} pts) — Top-{args.top_k} per material",
            fontsize=13
        )
        fig2.tight_layout()
        summary_path = out_dir / "summary_comparison.png"
        fig2.savefig(summary_path, bbox_inches="tight")
        plt.close(fig2)
        print(f"\nSaved summary comparison → {summary_path}")

        # ---- JSON metrics ----
        json_path = out_dir / "fine_validation_metrics.json"
        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved metrics → {json_path}")

        # ---- console summary ----
        print("\n" + "=" * 72)
        print(f"{'Material':<20} {'Rank':>4}  "
              f"{'DS AvgAbs':>10}  {'Fine AvgAbs':>11}  {'ΔAvgAbs':>8}  "
              f"{'DS Jsc':>8}  {'Fine Jsc':>9}  {'ΔJsc':>7}")
        print("-" * 72)
        for m in all_metrics:
            print(
                f"{m['material']:<20} {m['rank']:>4}  "
                f"{m['dataset_avg_abs']:>10.4f}  {m['fine_avg_abs']:>11.4f}  "
                f"{m['delta_avg_abs']:>+8.4f}  "
                f"{m['dataset_jsc']:>8.4f}  {m['fine_jsc']:>9.4f}  "
                f"{m['delta_jsc']:>+7.4f}"
            )
        print("=" * 72)


if __name__ == "__main__":
    main()
