#!/usr/bin/env python
"""
Find and visualize the distribution of 2D dataset samples.
Plots a histogram of spectrum-averaged absorptances and a scatter plot of wavelength vs. absorptance colored by wavelength.

Usage:
    python Scripts/evaluate_2d_dataset_baseline.py --data_dir Data/LHS_Dataset_Si
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Styling
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# Per-material colors — match the matplotlib default cycle order used in
# evaluate_dataset_baseline.py (Si=C0, TiO2=C1, Si3N4=C2)
MATERIAL_COLORS = {
    "Si":    "#1f77b4",  # tab:blue
    "TiO2":  "#ff7f0e",  # tab:orange
    "Si3N4": "#2ca02c",  # tab:green
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate 2D dataset baseline statistics and distributions.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to a specific 2D dataset directory. If None, runs on all LHS_Dataset_* directories found in Data/")
    p.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                   help="Which split of the dataset to evaluate (default: both)")
    p.add_argument("--target_key", type=str, default="A_film_normal",
                   choices=["A_film_normal", "A_film_oblique", "A_grating_normal", "A_grating_oblique"],
                   help="Dataset key to use for absorptance (default: A_film_normal)")
    p.add_argument("--h_val", nargs="+", type=float, help="Target height in nm, or range (min max)")
    p.add_argument("--h_tolerance", type=float, default=0.5, help="Tolerance for height matching")
    p.add_argument("--inc_val", type=float, default=None, help="Target incident angle in degrees (e.g., 0.0 for normal)")
    p.add_argument("--inc_tolerance", type=float, default=0.5, help="Tolerance for matching inc_val (degrees)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save the plots. Defaults to '<data_dir>/evaluation'")
    return p.parse_args()

def get_folder_name(args) -> str:
    parts = []
    if args.h_val is not None:
        if len(args.h_val) == 2:
            parts.append(f"h{int(args.h_val[0])}-{int(args.h_val[1])}")
        else:
            parts.append(f"h{int(args.h_val[0])}_tol{args.h_tolerance}")
    if hasattr(args, "inc_val") and args.inc_val is not None:
        parts.append(f"inc{args.inc_val}_tol{args.inc_tolerance}")
    return "_".join(parts) if parts else "all_data"

def get_plot_title_suffix(args) -> str:
    parts = []
    if args.h_val is not None:
        if len(args.h_val) == 2:
            parts.append(f"height: {int(args.h_val[0])}-{int(args.h_val[1])} nm")
        else:
            parts.append(f"height: {int(args.h_val[0])}±{args.h_tolerance} nm")
    if hasattr(args, "inc_val") and args.inc_val is not None:
        parts.append(f"inc_ang: {args.inc_val}°")
    elif args.h_val is not None and "oblique" not in getattr(args, "target_key", ""):
        parts.append("incident angle: 0°")
    return " | ".join(parts)

def analyze_dataset_folder(data_path: Path, args):
    print("=" * 80)
    print(f"Analyzing 2D dataset folder: {data_path.name}")
    print("=" * 80)
    
    # Extract material name from directory
    material_name = data_path.name.replace("LHS_Dataset_", "")
    
    prefix = "A_film" if "film" in args.target_key else "A_grating"
    target_name = prefix if args.inc_val is None else args.target_key
    
    # Find files to load
    files_to_load = []
    if args.split in ("train", "both"):
        t_file = data_path / "train_dataset.pt"
        if t_file.exists():
            files_to_load.append(("train", t_file))
        else:
            print(f"Warning: train dataset not found at {t_file}")
            
    if args.split in ("val", "both"):
        v_file = data_path / "val_dataset.pt"
        if v_file.exists():
            files_to_load.append(("val", v_file))
        else:
            print(f"Warning: val dataset not found at {v_file}")
            
    if not files_to_load:
        print(f"Error: No dataset files found in {data_path} for split '{args.split}'")
        return
        
    all_targets = []
    all_hs = []
    all_inc_angs = []
    all_params_x = []
    
    for split_name, file_path in files_to_load:
        print(f"Loading {split_name} split from: {file_path.name} ...")
        d = torch.load(file_path, map_location="cpu", weights_only=False)
        
        prefix = "A_film" if "film" in args.target_key else "A_grating"
        
        A_list = []
        inc_list = []
        
        # Determine which keys to append
        if args.inc_val is None or args.inc_val <= 1e-3:
            norm_key = f"{prefix}_normal"
            if norm_key in d:
                A_list.append((norm_key, d[norm_key].float()))
                inc_list.append(torch.zeros_like(d["h"].float()))
                
        if args.inc_val is None or (args.inc_val is not None and args.inc_val > 1e-3):
            obl_key = f"{prefix}_oblique"
            if obl_key in d:
                A_list.append((obl_key, d[obl_key].float()))
                inc_list.append(d["inc_ang"].float() if "inc_ang" in d else torch.zeros_like(d["h"].float()))
                
        for key_name, tgt in A_list:
            # Simple sanity filter (absorptance must be within physical ranges)
            # Check max/min across all wavelengths and polarizations
            valid = (tgt.max(dim=2).values.max(dim=1).values <= 1.05) & (tgt.min(dim=2).values.min(dim=1).values >= -0.05)
            
            # Apply height filtering
            h = d["h"].float()
            if args.h_val is not None:
                if len(args.h_val) == 2:
                    valid &= (h >= args.h_val[0]) & (h <= args.h_val[1])
                else:
                    h_target = args.h_val[0]
                    valid &= (torch.abs(h - h_target) <= args.h_tolerance)
                    
            # Extract incident angle for this particular target (e.g. oblique or normal)
            if "oblique" in key_name:
                inc_ang = d["inc_ang"].float() if "inc_ang" in d else torch.zeros_like(h)
            else:
                inc_ang = torch.zeros_like(h)
                
            # Apply incident angle filtering
            if args.inc_val is not None:
                valid &= (torch.abs(inc_ang - args.inc_val) <= args.inc_tolerance)
                
            all_targets.append(tgt[valid])
            all_hs.append(h[valid])
            all_inc_angs.append(inc_ang[valid])
            if "params_x" in d:
                all_params_x.append(d["params_x"].float()[valid])
            
    if not all_targets:
        print("No valid samples loaded.")
        return
        
    targets = torch.cat(all_targets) # (N, wl_len, 2)
    hs = torch.cat(all_hs).numpy()
    inc_angs = torch.cat(all_inc_angs).numpy()
    
    if all_params_x:
        params_x = torch.cat(all_params_x).numpy()
    else:
        params_x = None
        
    N, wl_len, _ = targets.shape
    print(f"Successfully loaded {N} valid samples with {wl_len} wavelengths each.")
    
    # Compute wavelength grid
    WAVELENGTHS = np.linspace(300, 1100, wl_len)
    
    # Compute polarization average: (P + S) / 2
    avg_abs_spectrum = ((targets[:, :, 0] + targets[:, :, 1]) / 2.0).numpy() # (N, wl_len)
    
    # Compute spectrum-averaged absorptance per structure: (N,)
    spectrum_averaged_abs = np.mean(avg_abs_spectrum, axis=1)
    
    # Calculate statistics
    mean_val = np.mean(spectrum_averaged_abs)
    std_val = np.std(spectrum_averaged_abs)
    min_val = np.min(spectrum_averaged_abs)
    max_val = np.max(spectrum_averaged_abs)
    
    print("\n--- Statistics (Spectrum-Averaged Absorptance) ---")
    print(f"Mean Absorptance: {mean_val:.4f}")
    print(f"Std Dev:          {std_val:.4f}")
    print(f"Minimum:          {min_val:.4f}")
    print(f"Maximum:          {max_val:.4f}")
    
    percentiles = [25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(spectrum_averaged_abs, percentiles)
    for p, v in zip(percentiles, perc_vals):
        print(f"{p}th percentile:   {v:.4f}")
        
    # Find the top structures
    top_indices = np.argsort(spectrum_averaged_abs)[::-1][:5]
    print(f"\n--- Top 5 Performing Structures ({target_name}) ---")
    print(f"{'Rank':<6}{'Avg Abs':<12}{'Height (nm)':<14}{'Total Grat H (nm)':<20}")
    for rank, idx in enumerate(top_indices):
        h_val = hs[idx]
        abs_val = spectrum_averaged_abs[idx]
        
        tot_grat_h = 2.0 * np.sum(params_x[idx, :, 0]) if params_x is not None else 0.0
        
        print(f"{rank+1:<6}{abs_val:<12.4f}{h_val:<14.2f}{tot_grat_h:<20.2f}")
        
    # Setup output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = data_path / "evaluation" / get_folder_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting
    mat_color = MATERIAL_COLORS.get(material_name, "#1f77b4")
    title_suffix = f"\n({get_plot_title_suffix(args)})" if get_plot_title_suffix(args) else ""
    folder_suffix = get_folder_name(args)

    # --- Figure 1: Histogram of spectrum-averaged absorptances ---
    fig_hist, ax1 = plt.subplots(figsize=(8, 6))
    counts, bins, patches = ax1.hist(spectrum_averaged_abs, bins=50, edgecolor='black', alpha=0.7, color=mat_color)
    ax1.set_title(f"Spectrum-Averaged Absorptance\n({material_name}){title_suffix}")
    ax1.set_xlabel(f"Average Absorptance ({target_name})")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    hist_name = f"dataset_baseline_2d_{target_name}_{args.split}_{folder_suffix}_hist.png"
    hist_path = out_dir / hist_name
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved histogram to: {hist_path}")

    # --- Figure 2: Spectral Response Density scatter ---
    wls_scatter = np.tile(WAVELENGTHS, N)
    abs_scatter = avg_abs_spectrum.flatten()
    fig_sc, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(wls_scatter, abs_scatter, c=wls_scatter, cmap='turbo', s=1, alpha=0.15, edgecolors='none', rasterized=True)
    cbar = fig_sc.colorbar(sc, ax=ax2)
    cbar.set_label("Wavelength (nm)")
    ax2.set_title(f"Spectral Response Density\n({material_name}){title_suffix}")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel(f"Absorptance ({target_name})")
    ax2.set_xlim(280, 1120)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    scatter_name = f"dataset_baseline_2d_{target_name}_{args.split}_{folder_suffix}_scatter.png"
    scatter_path = out_dir / scatter_name
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"Saved spectral scatter to: {scatter_path}")
    
    # ------------------ Plotting Jsc ------------------
    # Calculate Jsc
    from Utils.utils import sun_weights, get_jsc_scaling_factor
    wls_t = torch.tensor(WAVELENGTHS, dtype=torch.float32)
    photon_flux = sun_weights(wls_t) * wls_t
    
    # Calculate Jsc for P and S polarizations
    jsc_p = (targets[:, :, 0] * photon_flux.unsqueeze(0)).sum(dim=1)
    jsc_s = (targets[:, :, 1] * photon_flux.unsqueeze(0)).sum(dim=1)
    jsc = ((jsc_p + jsc_s) / 2.0) * get_jsc_scaling_factor(wl_len)
    jsc = jsc.numpy()
    
    # Cosine correction
    cos_theta = np.cos(inc_angs * np.pi / 180.0)
    jsc = jsc * cos_theta
    
    # Calculate Jsc statistics
    jsc_mean = np.mean(jsc)
    jsc_std = np.std(jsc)
    jsc_min = np.min(jsc)
    jsc_max = np.max(jsc)
    
    print("\n--- Statistics (Short-Circuit Current Jsc, mA/cm2) ---")
    print(f"Mean Jsc:    {jsc_mean:.4f}")
    print(f"Std Dev:     {jsc_std:.4f}")
    print(f"Minimum:     {jsc_min:.4f}")
    print(f"Maximum:     {jsc_max:.4f}")
    
    jsc_perc_vals = np.percentile(jsc, percentiles)
    for p, v in zip(percentiles, jsc_perc_vals):
        print(f"{p}th percentile:   {v:.4f}")
        
    # Find the top structures by Jsc
    top_jsc_indices = np.argsort(jsc)[::-1][:5]
    print(f"\n--- Top 5 Performing Structures by Jsc ({target_name}) ---")
    print(f"{'Rank':<6}{'Jsc (mA/cm2)':<16}{'Height (nm)':<14}{'Total Grat H (nm)':<20}")
    for rank, idx in enumerate(top_jsc_indices):
        h_val = hs[idx]
        jsc_val = jsc[idx]
        tot_grat_h = 2.0 * np.sum(params_x[idx, :, 0]) if params_x is not None else 0.0
        print(f"{rank+1:<6}{jsc_val:<16.4f}{h_val:<14.2f}{tot_grat_h:<20.2f}")
        
    # --- Jsc Figure 1: Histogram ---
    fig_jsc_hist, ax1_j = plt.subplots(figsize=(8, 6))
    ax1_j.hist(jsc, bins=50, edgecolor='black', alpha=0.7, color=mat_color)
    ax1_j.set_title(f"Short-Circuit Current $J_{{sc}}$ Distribution\n({material_name}){title_suffix}")
    ax1_j.set_xlabel(r"Short-Circuit Current $J_{sc}$ (mA/cm$^2$)")
    ax1_j.set_ylabel("Count")
    ax1_j.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    jsc_hist_name = f"dataset_baseline_2d_jsc_{target_name}_{args.split}_{folder_suffix}_hist.png"
    jsc_hist_path = out_dir / jsc_hist_name
    plt.savefig(jsc_hist_path, dpi=200)
    plt.close()
    print(f"Saved Jsc histogram to: {jsc_hist_path}")

    # --- Jsc Figure 2: Scatter Plot of Film Height vs Jsc ---
    fig_jsc_sc, ax2_j = plt.subplots(figsize=(8, 6))
    if params_x is not None:
        tot_grat_hs = 2.0 * np.sum(params_x[:, :, 0], axis=1)
        # Sort by tot_grat_hs ascending so high values are drawn last (on top)
        sort_idx = np.argsort(tot_grat_hs)
        sc_j = ax2_j.scatter(hs[sort_idx], jsc[sort_idx], c=tot_grat_hs[sort_idx], cmap='viridis', s=15, alpha=0.7, edgecolors='none')
        cbar_j = fig_jsc_sc.colorbar(sc_j, ax=ax2_j)
        cbar_j.set_label("Total Grating Height (nm)")
    else:
        ax2_j.scatter(hs, jsc, color=mat_color, s=15, alpha=0.6, edgecolors='none')
    ax2_j.set_title(f"$J_{{sc}}$ vs Film Height\n({material_name}){title_suffix}")
    ax2_j.set_xlabel("Film Height (nm)")
    ax2_j.set_ylabel(r"Short-Circuit Current $J_{sc}$ (mA/cm$^2$)")
    ax2_j.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    jsc_scatter_name = f"dataset_baseline_2d_jsc_{target_name}_{args.split}_{folder_suffix}_scatter.png"
    jsc_scatter_path = out_dir / jsc_scatter_name
    plt.savefig(jsc_scatter_path, dpi=200)
    plt.close()
    print(f"Saved Jsc scatter to: {jsc_scatter_path}")
    print("=" * 80 + "\n")

def main():
    args = parse_args()
    
    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Specified data directory {data_path} does not exist.")
            return
        analyze_dataset_folder(data_path, args)
    else:
        # Scan Data directory for LHS_Dataset_* but exclude LHS_3D_Dataset_*
        data_root = PROJECT_ROOT / "Data"
        all_folders = sorted(list(data_root.glob("LHS_Dataset_*")))
        dataset_folders = [f for f in all_folders if f.is_dir() and "3D" not in f.name]
        
        if not dataset_folders:
            print("Error: No 2D LHS_Dataset_* directories found in Data/")
            return
            
        print(f"Found {len(dataset_folders)} 2D dataset folders in Data/. Running evaluation for all...")
        for folder in dataset_folders:
            analyze_dataset_folder(folder, args)

if __name__ == "__main__":
    main()
