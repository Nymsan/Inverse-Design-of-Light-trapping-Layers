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
    p.add_argument("--h_val", type=float, help="Target height in nm")
    p.add_argument("--h_tolerance", type=float, default=0.5, help="Tolerance for height matching")
    p.add_argument("--inc_val", type=float, default=None, help="Target incident angle in degrees (e.g., 0.0 for normal)")
    p.add_argument("--inc_tolerance", type=float, default=0.5, help="Tolerance for incident angle matching")
    return p.parse_args()

def get_dataset_baseline(ckpt_dir: Path, bands=None, h_val=None, h_tolerance=0.5, inc_val=None, inc_tolerance=0.5) -> dict:
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
    
    for mat_name, relative_path in stats["materials"].items():
        mat_dir = PROJECT_ROOT / "Data" / Path(relative_path).name
        all_targets = []
        all_geos = []
        
        batch_files = sorted(list(mat_dir.glob("batch_*.pt")))
        for bf in batch_files:
            if "batch_9" in bf.name: continue
            data = torch.load(bf, map_location="cpu", weights_only=False)
            
            params = data["params_x"]
            h = data["h"]
            inc_ang = data["inc_ang"]
            
            # Use normal incidence if explicitly requested, otherwise use oblique and filter
            if inc_val is not None and inc_val == 0.0:
                A_film = data["A_film_normal"]
                inc_ang_eff = torch.zeros_like(inc_ang)
            else:
                A_film = data["A_film_oblique"]
                inc_ang_eff = inc_ang
                
            batch_targets = torch.cat([A_film[:, :, 0], A_film[:, :, 1]], dim=1)
            
            valid_mask = (batch_targets.max(dim=1).values <= 1.05)
            if h_val is not None:
                valid_mask &= (torch.abs(h - h_val) <= h_tolerance)
            if inc_val is not None and inc_val != 0.0:
                valid_mask &= (torch.abs(inc_ang - inc_val) <= inc_tolerance)
            
            if not valid_mask.any():
                continue

            all_targets.append(batch_targets[valid_mask])
            amps = params[valid_mask, :, 0]
            phases = params[valid_mask, :, 1]
            h_valid = h[valid_mask].unsqueeze(1)
            inc_valid = inc_ang_eff[valid_mask].unsqueeze(1)
            geo = torch.cat([amps, phases, h_valid, inc_valid], dim=1)
            all_geos.append(geo)
            
        if not all_targets:
            continue
            
        targets = torch.cat(all_targets, dim=0)
        geos = torch.cat(all_geos, dim=0)
        
        num_matched = len(targets)
        h_str = f"h={h_val}±{h_tolerance}nm" if h_val is not None else "all heights"
        band_str = "matched bands" if bands else "full spectrum"
        print(f"[{mat_name}] Dataset Baseline: Found {num_matched} valid structures for {h_str} in {band_str}.")
        
        p_pol = targets[:, :len(WAVELENGTHS)]
        s_pol = targets[:, len(WAVELENGTHS):]
        avg_p = p_pol[:, mask_dataset].mean(dim=1)
        avg_s = s_pol[:, mask_dataset].mean(dim=1)
        avg_abs = (avg_p + avg_s) / 2.0
        
        valid_mask = (targets.max(dim=1).values <= 1.05)
        avg_abs[~valid_mask] = -1.0
        
        results[mat_name] = {
            "targets": targets,
            "geos": geos,
            "avg_abs": avg_abs
        }
    return results, stats

def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    
    out_dir = ckpt_dir / "evaluation" / "dataset_baseline"
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
    results, stats = get_dataset_baseline(ckpt_dir, bands=bands, h_val=args.h_val, h_tolerance=args.h_tolerance, inc_val=args.inc_val, inc_tolerance=args.inc_tolerance)
    n_harmonics = stats["n_harmonics"]
    n_wavelengths = stats["n_wavelengths"]
    WAVELENGTHS = np.linspace(300, 1100, n_wavelengths // 2)
    
    for mat_name, res in results.items():
        print(f"\nPlotting baseline for material: {mat_name}")
        targets = res["targets"]
        geos = res["geos"]
        avg_abs = res["avg_abs"]
        # Valid structures have avg_abs >= 0
        valid_idx = torch.where(avg_abs >= 0)[0]
        if len(valid_idx) == 0:
            print(f"No valid structures found for {mat_name}")
            continue
            
        valid_abs = avg_abs[valid_idx]
        
        # Determine best and worst indices
        k = min(args.top_k, len(valid_idx))
        best_local_indices = torch.topk(valid_abs, k, largest=True).indices
        worst_local_indices = torch.topk(valid_abs, k, largest=False).indices
        
        best_indices = valid_idx[best_local_indices]
        worst_indices = valid_idx[worst_local_indices]
        
        for plot_mode, indices in [("best", best_indices), ("worst", worst_indices)]:
            fig, axes = plt.subplots(k, 4, figsize=(24, 5 * k), squeeze=False, layout="constrained")
            
            for i, idx in enumerate(indices):
                target = targets[idx]
                geo = geos[idx]
                score = avg_abs[idx].item()
                
                # Reconstruct profile
                amps = geo[:n_harmonics].numpy()
                phases = geo[n_harmonics:2*n_harmonics].numpy()
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
                ax_p.set_title(f"{mode_str} Rank {i+1} (Avg Abs: {score:.3f}) - P-Pol", fontsize=14)
                ax_p.set_ylabel("Absorptance")
                
                # Plot S-Pol
                ax_s = axes[i, 1]
                ax_s.plot(WAVELENGTHS, target[n_wavelengths//2:].numpy(), "k-", lw=2)
                ax_s.set_ylim(-0.05, 1.05)
                if bands:
                    for bmin, bmax in bands:
                        ax_s.axvspan(bmin, bmax, color="gray", alpha=0.2)
                ax_s.set_title(f"{mode_str} Rank {i+1} (Avg Abs: {score:.3f}) - S-Pol", fontsize=14)
                
                # Structure Cross Section
                ax_xs = axes[i, 2]
                ax_xs.plot(r_grid, prof, "k-", lw=2)
                ax_xs.fill_between(r_grid, 0, prof, color="tab:blue", alpha=0.3)
                ax_xs.set_title(f"Structure (h={h_nm:.0f}nm, inc={inc_ang:.1f}°)", fontsize=14)
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
            bands_str = "_".join([f"{int(b[0])}-{int(b[1])}" for b in bands]) if bands else "full_spectrum"
            suffix = "_worst" if plot_mode == "worst" else ""
            save_path = out_dir / f"baseline{suffix}_{mat_name}_{bands_str}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {plot_mode} plot to {save_path}")

if __name__ == "__main__":
    main()
