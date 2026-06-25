#!/usr/bin/env python
"""
Find and visualize the top-performing structures directly from the dataset.
Provides a strong baseline for model performance.

Usage:
    python Scripts/evaluate_dataset_baseline.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import build_profile, MATERIAL_LIBRARY

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    p.add_argument("--top_k", type=int, default=1, help="Number of top structures to show per material")
    p.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to evaluate, e.g., --bands 500 750 800 900")
    p.add_argument("--h_val", nargs="+", type=float, help="Target height in nm, or range (min max)")
    p.add_argument("--h_tolerance", type=float, default=0.5, help="Tolerance for height matching")
    p.add_argument("--inc_val", type=float, default=None, help="Target incident angle in degrees (e.g., 0.0 for normal)")
    p.add_argument("--inc_tolerance", type=float, default=0.5, help="Tolerance for matching inc_val (degrees)")
    p.add_argument("--optimize_jsc", action="store_true", help="Optimize Short-Circuit Current (Jsc) weighted by AM1.5G photon flux and cos(inc_ang), rather than average absorptance.")
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
        
    if getattr(args, "optimize_jsc", False):
        parts.append("jsc")
        
    return "_".join(parts) if parts else "all_data"

def get_dataset_baseline(ckpt_dir: Path, bands=None, h_val=None, h_tolerance=0.5, inc_val=None, inc_tolerance=0.5, optimize_jsc=False) -> dict:
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found at {stats_path}")
        
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    n_wavelengths = stats["n_wavelengths"]
    WAVELENGTHS = np.linspace(300, 1100, n_wavelengths // 2)
    
    if bands:
        mask_dataset = np.zeros(len(WAVELENGTHS), dtype=bool)
        for bmin, bmax in bands:
            mask_dataset |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
    else:
        mask_dataset = np.ones(len(WAVELENGTHS), dtype=bool)

    results = {}
    
    for mat_name in stats["materials"]:
        # Reconstruct the original data directory path
        prefix = stats.get("dataset_prefixes", ["LHS_Dataset"])[0]
        mat_dir = PROJECT_ROOT / "Data" / f"{prefix}_{mat_name}"
        all_targets = []
        all_geos = []
        
        files_to_load = []
        t_file = mat_dir / "train_dataset.pt"
        v_file = mat_dir / "val_dataset.pt"
        if t_file.exists(): files_to_load.append(t_file)
        if v_file.exists(): files_to_load.append(v_file)
            
        for bf in files_to_load:
            data = torch.load(bf, map_location="cpu", weights_only=False)
            
            params = data["params_x"]
            h = data["h"]
            inc_ang = data["inc_ang"]
            
            A_list = []
            inc_list = []
            
            # If no inc_val is provided, we want BOTH normal and oblique data.
            # If inc_val == 0.0, we want ONLY normal.
            # If inc_val > 0.0, we want ONLY oblique.
            if inc_val is None or inc_val <= 1e-3:
                if "A_film_normal" in data:
                    A_list.append(data["A_film_normal"])
                    inc_list.append(torch.zeros_like(inc_ang))
            if inc_val is None or (inc_val is not None and inc_val > 1e-3):
                if "A_film_oblique" in data:
                    A_list.append(data["A_film_oblique"])
                    inc_list.append(inc_ang)
                    
            for A_film, inc_ang_eff in zip(A_list, inc_list):
                batch_targets = torch.cat([A_film[:, :, 0], A_film[:, :, 1]], dim=1)
                
                valid_mask = (batch_targets.max(dim=1).values <= 1.05)
                if h_val is not None:
                    if isinstance(h_val, list) and len(h_val) == 2:
                        valid_mask &= (h >= h_val[0]) & (h <= h_val[1])
                    else:
                        h_target = h_val[0] if isinstance(h_val, list) else h_val
                        valid_mask &= (torch.abs(h - h_target) <= h_tolerance)
                if inc_val is not None and inc_val > 1e-3:
                    valid_mask &= (torch.abs(inc_ang_eff - inc_val) <= inc_tolerance)
                
                if not valid_mask.any():
                    continue

                all_targets.append(batch_targets[valid_mask])
                px = params[valid_mask]
                px_flat = px.view(px.shape[0], -1)
                h_valid = h[valid_mask].unsqueeze(1)
                inc_valid = inc_ang_eff[valid_mask].unsqueeze(1)
                geo = torch.cat([px_flat, h_valid, inc_valid], dim=1)
                all_geos.append(geo)
            
        if not all_targets:
            continue
            
        targets = torch.cat(all_targets, dim=0)
        geos = torch.cat(all_geos, dim=0)
        
        num_matched = len(targets)
        if h_val is not None:
            if isinstance(h_val, list) and len(h_val) == 2:
                h_str = f"h=[{h_val[0]}, {h_val[1]}]nm"
            else:
                h_target = h_val[0] if isinstance(h_val, list) else h_val
                h_str = f"h={h_target}±{h_tolerance}nm"
        else:
            h_str = "all heights"
        if inc_val is not None:
            inc_str = f"inc={inc_val}±{inc_tolerance}°"
        else:
            inc_str = "all angles"
        band_str = "matched bands" if bands else "full spectrum"
        print(f"[{mat_name}] Filtered dataset down to {num_matched} valid samples for {h_str}, {inc_str}, in {band_str}.")
        
        p_pol = targets[:, :len(WAVELENGTHS)]
        s_pol = targets[:, len(WAVELENGTHS):]
        avg_p = p_pol[:, mask_dataset].mean(dim=1)
        avg_s = s_pol[:, mask_dataset].mean(dim=1)
        avg_abs = (avg_p + avg_s) / 2.0
        
        if optimize_jsc:
            from Utils.utils import sun_weights, get_jsc_scaling_factor
            wls_t = torch.tensor(WAVELENGTHS, dtype=torch.float32)
            photon_flux = sun_weights(wls_t) * wls_t
            mask_t = torch.tensor(mask_dataset, dtype=torch.float32)
            
            jsc_p = (p_pol * mask_t * photon_flux.unsqueeze(0)).sum(dim=1)
            jsc_s = (s_pol * mask_t * photon_flux.unsqueeze(0)).sum(dim=1)
            metric = (jsc_p + jsc_s) / 2.0
            metric = metric * get_jsc_scaling_factor(len(WAVELENGTHS))
            
            inc_ang_deg = geos[:, -1]
            cos_theta = torch.cos(inc_ang_deg * torch.pi / 180.0)
            metric = metric * cos_theta
        else:
            metric = avg_abs
        
        valid_mask = (targets.max(dim=1).values <= 1.05)
        metric[~valid_mask] = -1.0
        
        results[mat_name] = {
            "targets": targets,
            "geos": geos,
            "metric": metric,
            "avg_abs": avg_abs
        }
    return results, stats

def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    
    out_dir = ckpt_dir / "evaluation" / "dataset_baseline" / get_folder_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse bands
    bands = []
    if args.bands:
        if len(args.bands) % 2 != 0:
            print("Error: --bands must have an even number of arguments (min max pairs)")
            return
        for i in range(0, len(args.bands), 2):
            bands.append((args.bands[i], args.bands[i+1]))
            
    # We will evaluate based on the provided bands
    results, stats = get_dataset_baseline(ckpt_dir, bands=bands, h_val=args.h_val, h_tolerance=args.h_tolerance, inc_val=args.inc_val, inc_tolerance=args.inc_tolerance, optimize_jsc=args.optimize_jsc)
    n_harmonics = stats["n_harmonics"]
    n_wavelengths = stats["n_wavelengths"]
    WAVELENGTHS = np.linspace(300, 1100, n_wavelengths // 2)
    
    global_min, global_max = 1000.0, 0.0
    for res in results.values():
        valid_idx = torch.where(res["metric"] >= 0)[0]
        if len(valid_idx) > 0:
            global_min = min(global_min, res["metric"][valid_idx].min().item())
            global_max = max(global_max, res["metric"][valid_idx].max().item())
            
    bin_width = (global_max - global_min) / 50.0 if global_max > global_min else 0.01
    if bin_width == 0: bin_width = 0.01
    
    grid_min = np.floor(global_min / bin_width) * bin_width
    grid_max = np.ceil(global_max / bin_width) * bin_width
    num_bins = max(1, int(np.round((grid_max - grid_min) / bin_width)))
    bin_edges = np.linspace(grid_min, grid_max, num_bins + 1)
    
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    for mat_name, res in results.items():
        metric_vals = res["metric"]
        valid_idx = torch.where(metric_vals >= 0)[0]
        if len(valid_idx) == 0:
            continue
        valid_metric = metric_vals[valid_idx].numpy()
        ax_hist.hist(valid_metric, bins=bin_edges, alpha=0.5, label=mat_name)
    ax_hist.set_xlabel("Short-Circuit Current (mA/cm2, pseudo)" if args.optimize_jsc else "Average Absorptance")
    ax_hist.set_ylabel("Count")
    ax_hist.set_yscale("log")
    
    title_parts = []
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            title_parts.append(f"h=[{args.h_val[0]},{args.h_val[1]}]nm")
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            title_parts.append(f"h={h_target}±{args.h_tolerance}nm")
    if args.inc_val is not None:
        title_parts.append(f"inc={args.inc_val}±{args.inc_tolerance}°")
    if args.bands:
        title_parts.append("Matched Bands")
        
    title_str = " | ".join(title_parts) if title_parts else "All Geometries, All Angles"
    ax_hist.set_title(f"Dataset Metric Distribution\n({title_str})", fontsize=11)
    ax_hist.legend()
    hist_path = out_dir / "dataset_metric_histogram.png"
    fig_hist.savefig(hist_path)
    plt.close(fig_hist)
    print(f"Saved dataset histogram to {hist_path}")
    
    for mat_name, res in results.items():
        print(f"\nPlotting baseline for material: {mat_name}")
        targets = res["targets"]
        geos = res["geos"]
        metric = res["metric"]
        # Valid structures have metric >= 0
        valid_idx = torch.where(metric >= 0)[0]
        if len(valid_idx) == 0:
            print(f"No valid structures found for {mat_name}")
            continue
            
        valid_met = metric[valid_idx]
        
        # Determine best and worst indices
        k = min(args.top_k, len(valid_idx))
        best_local_indices = torch.topk(valid_met, k, largest=True).indices
        worst_local_indices = torch.topk(valid_met, k, largest=False).indices
        
        best_indices = valid_idx[best_local_indices]
        worst_indices = valid_idx[worst_local_indices]
        
        for plot_mode, indices in [("best", best_indices), ("worst", worst_indices)]:
            fig, axes = plt.subplots(k, 4, figsize=(24, 5 * k), squeeze=False, layout="constrained")
            
            for i, idx in enumerate(indices):
                target = targets[idx]
                geo = geos[idx]
                score = metric[idx].item()
                
                # Reconstruct profile
                amps = geo[0:2*n_harmonics:2].numpy()
                phases = geo[1:2*n_harmonics:2].numpy()
                h_nm = geo[-2].item()
                inc_ang = geo[-1].item()
                
                r_grid = np.linspace(0, 1000, 128)
                harmonic_idx = np.arange(1, n_harmonics + 1)
                
                grating_height = 2.0 * amps.sum() + 1e-9
                arg = 2.0 * np.pi * harmonic_idx[:, None] * r_grid[None, :] / 1000.0 - phases[:, None]
                cosines = amps[:, None] * np.cos(arg)
                prof = grating_height / 2.0 + cosines.sum(axis=0)
                
                # Plot P-Pol
                ax_p = axes[i, 0]
                ax_p.plot(WAVELENGTHS, target[:n_wavelengths//2].numpy(), "k-", lw=2)
                ax_p.set_ylim(-0.05, 1.05)
                if bands:
                    for bmin, bmax in bands:
                        ax_p.axvspan(bmin, bmax, color="gray", alpha=0.2)
                mode_str = "Best" if plot_mode == "best" else "Worst"
                ax_p.set_title(f"{mode_str} Rank {i+1} (Score: {score:.3f}) - P-Pol", fontsize=14)
                ax_p.set_ylabel("Absorptance")
                
                # Plot S-Pol
                ax_s = axes[i, 1]
                ax_s.plot(WAVELENGTHS, target[n_wavelengths//2:].numpy(), "k-", lw=2)
                ax_s.set_ylim(-0.05, 1.05)
                if bands:
                    for bmin, bmax in bands:
                        ax_s.axvspan(bmin, bmax, color="gray", alpha=0.2)
                ax_s.set_title(f"{mode_str} Rank {i+1} (Score: {score:.3f}) - S-Pol", fontsize=14)
                
                # Structure Cross Section
                ax_xs = axes[i, 2]
                ax_xs.plot(r_grid, prof, "k-", lw=2)
                ax_xs.fill_between(r_grid, 0, prof, color="tab:blue", alpha=0.3)
                ax_xs.set_title(f"Structure (Film Height={h_nm:.0f}nm, Inc Ang={inc_ang:.1f}°)", fontsize=14)
                ax_xs.set_ylim(0, max(120, grating_height * 1.2))
                ax_xs.set_xlim(0, 1000)
                ax_xs.set_ylabel("Thickness (nm)")
                
                # Fourier parameters
                cmap = plt.cm.viridis
                ax_h = axes[i, 3]
                x_pos = np.arange(1, n_harmonics + 1)
                ax_h.bar(x_pos, amps, color=cmap(0.5), edgecolor="black")
                ax_h.set_ylabel("Amplitude (nm)", color=cmap(0.5))
                ax_h.tick_params(axis='y', labelcolor=cmap(0.5))
                ax_p2 = ax_h.twinx()
                ax_p2.plot(x_pos, phases, 'o', color=cmap(0.9), markersize=8)
                ax_p2.set_ylabel("Phase (rad)", color=cmap(0.9))
                ax_p2.tick_params(axis='y', labelcolor=cmap(0.9))
                ax_h.set_title("Harmonics Amplitudes & Phases", fontsize=14)
                ax_h.set_xticks(x_pos)
                
                if i == k - 1:
                    ax_p.set_xlabel("Wavelength (nm)")
                    ax_s.set_xlabel("Wavelength (nm)")
                    ax_xs.set_xlabel("x (nm)")
                    ax_h.set_xlabel("Harmonic Index")

            plt.suptitle(f"{mode_str} {k} Structure(s) in Dataset: {mat_name}", fontsize=18)
            save_path = out_dir / f"dataset_baseline_{'worst_' if plot_mode == 'worst' else ''}{mat_name}.png"
            fig.savefig(save_path)
            plt.close()
            print(f"Saved {plot_mode} plot to {save_path}")

if __name__ == "__main__":
    main()
