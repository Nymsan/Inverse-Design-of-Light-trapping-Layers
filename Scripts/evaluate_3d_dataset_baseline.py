#!/usr/bin/env python
"""
Find and visualize the distribution of 3D dataset samples.
Plots a histogram of absorptances and a scatter plot of wavelength vs. absorptance colored by wavelength.

Usage:
    python Scripts/evaluate_3d_dataset_baseline.py --data_dir Data/LHS_3D_Dataset_Si
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
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate 3D dataset baseline statistics and distributions.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to a specific 3D dataset directory. If None, runs on all LHS_3D_Dataset_* directories found in Data/")
    p.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                   help="Which split of the dataset to evaluate (default: both)")
    p.add_argument("--target_key", type=str, default="A_film_normal",
                   choices=["A_film_normal", "A_grating_normal"],
                   help="Dataset key to use for absorptance (default: A_film_normal)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save the plots. Defaults to '<data_dir>/evaluation'")
    return p.parse_args()

def analyze_dataset_folder(data_path: Path, args):
    print("=" * 80)
    print(f"Analyzing 3D dataset folder: {data_path.name}")
    print("=" * 80)
    
    # Extract material name from directory
    material_name = data_path.name.replace("LHS_3D_Dataset_", "")
    
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
        
    companion_key = "A_film_max_wl" if args.target_key == "A_film_normal" else "A_grating_max_wl"
    all_wls = []
    all_hs = []
    all_abs = []
    all_comp_abs = []
    all_params_x = []
    all_params_y = []
    
    for split_name, file_path in files_to_load:
        print(f"Loading {split_name} split from: {file_path.name} ...")
        d = torch.load(file_path, map_location="cpu", weights_only=False)
        
        tgt = d[args.target_key].float()  # (B, 2)
        comp_tgt = d[companion_key].float() if companion_key in d else None
        
        # Compute polarization average
        avg_abs = (tgt[:, 0] + tgt[:, 1]) / 2.0
        if comp_tgt is not None:
            comp_avg_abs = (comp_tgt[:, 0] + comp_tgt[:, 1]) / 2.0
        else:
            comp_avg_abs = None
            
        # Simple sanity filter (absorptance must be within physical ranges)
        valid = (tgt.max(dim=1).values <= 1.05) & (tgt.min(dim=1).values >= -0.05)
        
        all_wls.append(d["wavelength"].float()[valid])
        all_hs.append(d["h"].float()[valid])
        all_abs.append(avg_abs[valid])
        if comp_avg_abs is not None:
            all_comp_abs.append(comp_avg_abs[valid])
        if "params_x" in d:
            all_params_x.append(d["params_x"].float()[valid])
        if "params_y" in d:
            all_params_y.append(d["params_y"].float()[valid])
            
    if not all_wls:
        print("No valid samples loaded.")
        return
        
    wls = torch.cat(all_wls).numpy()
    hs = torch.cat(all_hs).numpy()
    absorptances = torch.cat(all_abs).numpy()
    comp_absorptances = torch.cat(all_comp_abs).numpy() if all_comp_abs else None
    
    if all_params_x:
        params_x = torch.cat(all_params_x).numpy()
    else:
        params_x = None
        
    if all_params_y:
        params_y = torch.cat(all_params_y).numpy()
    else:
        params_y = None
        
    num_samples = len(absorptances)
    print(f"Successfully loaded {num_samples} valid samples.")
    
    # Calculate statistics
    mean_val = np.mean(absorptances)
    std_val = np.std(absorptances)
    min_val = np.min(absorptances)
    max_val = np.max(absorptances)
    
    print("\n--- Statistics (Randomized Wavelengths) ---")
    print(f"Mean Absorptance: {mean_val:.4f}")
    print(f"Std Dev:          {std_val:.4f}")
    print(f"Minimum:          {min_val:.4f}")
    print(f"Maximum:          {max_val:.4f}")
    
    percentiles = [25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(absorptances, percentiles)
    for p, v in zip(percentiles, perc_vals):
        print(f"{p}th percentile:   {v:.4f}")
        
    if comp_absorptances is not None:
        comp_mean = np.mean(comp_absorptances)
        comp_std = np.std(comp_absorptances)
        comp_min = np.min(comp_absorptances)
        comp_max = np.max(comp_absorptances)
        
        print("\n--- Statistics (Reference Wavelength: 495 nm) ---")
        print(f"Mean Absorptance: {comp_mean:.4f}")
        print(f"Std Dev:          {comp_std:.4f}")
        print(f"Minimum:          {comp_min:.4f}")
        print(f"Maximum:          {comp_max:.4f}")
        
        comp_perc_vals = np.percentile(comp_absorptances, percentiles)
        for p, v in zip(percentiles, comp_perc_vals):
            print(f"{p}th percentile:   {v:.4f}")
        
    # Find the top structures
    top_indices = np.argsort(absorptances)[::-1][:5]
    print(f"\n--- Top 5 Performing Structures ({args.target_key}) ---")
    print(f"{'Rank':<6}{'Absorptance':<14}{'Wavelength (nm)':<18}{'Height (nm)':<14}{'Max Amp X (nm)':<16}{'Max Amp Y (nm)':<16}")
    for rank, idx in enumerate(top_indices):
        wl_val = wls[idx]
        h_val = hs[idx]
        abs_val = absorptances[idx]
        
        max_amp_x = np.max(params_x[idx, :, 0]) if params_x is not None else 0.0
        max_amp_y = np.max(params_y[idx, :, 0]) if params_y is not None else 0.0
        
        print(f"{rank+1:<6}{abs_val:<14.4f}{wl_val:<18.2f}{h_val:<14.2f}{max_amp_x:<16.2f}{max_amp_y:<16.2f}")
        
    # Setup output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = data_path / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left: Histogram
    counts, bins, patches = ax1.hist(absorptances, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax1.set_title(f"Absorptance Distribution\n({material_name})")
    ax1.set_xlabel(f"Average Absorptance ({args.target_key})")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Right: Scatter Plot of Wavelength vs Absorptance colored by wavelength
    if comp_absorptances is not None:
        wls_scatter = np.concatenate([wls, np.full_like(wls, 495.0)])
        abs_scatter = np.concatenate([absorptances, comp_absorptances])
    else:
        wls_scatter = wls
        abs_scatter = absorptances
        
    sc = ax2.scatter(wls_scatter, abs_scatter, c=wls_scatter, cmap='turbo', s=15, alpha=0.6, edgecolors='none')
    
    # Add a clean colorbar
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Wavelength (nm)")
    
    ax2.set_title(f"Absorptance vs Wavelength\n({material_name})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel(f"Average Absorptance ({args.target_key})")
    ax2.set_xlim(280, 1120)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    plot_name = f"dataset_baseline_3d_{args.target_key}_{args.split}.png"
    save_path = out_dir / plot_name
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\nSaved plots to: {save_path}")
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
        # Scan Data directory for LHS_3D_Dataset_*
        data_root = PROJECT_ROOT / "Data"
        dataset_folders = sorted(list(data_root.glob("LHS_3D_Dataset_*")))
        if not dataset_folders:
            print("Error: No LHS_3D_Dataset_* directories found in Data/")
            return
            
        print(f"Found {len(dataset_folders)} 3D dataset folders in Data/. Running evaluation for all...")
        for folder in dataset_folders:
            if folder.is_dir():
                analyze_dataset_folder(folder, args)

if __name__ == "__main__":
    main()
