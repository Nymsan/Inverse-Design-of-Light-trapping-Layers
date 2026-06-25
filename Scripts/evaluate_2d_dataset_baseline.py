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
    p = argparse.ArgumentParser(description="Evaluate 2D dataset baseline statistics and distributions.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to a specific 2D dataset directory. If None, runs on all LHS_Dataset_* directories found in Data/")
    p.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                   help="Which split of the dataset to evaluate (default: both)")
    p.add_argument("--target_key", type=str, default="A_film_normal",
                   choices=["A_film_normal", "A_film_oblique", "A_grating_normal", "A_grating_oblique"],
                   help="Dataset key to use for absorptance (default: A_film_normal)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save the plots. Defaults to '<data_dir>/evaluation'")
    return p.parse_args()

def analyze_dataset_folder(data_path: Path, args):
    print("=" * 80)
    print(f"Analyzing 2D dataset folder: {data_path.name}")
    print("=" * 80)
    
    # Extract material name from directory
    material_name = data_path.name.replace("LHS_Dataset_", "")
    
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
    all_params_x = []
    
    for split_name, file_path in files_to_load:
        print(f"Loading {split_name} split from: {file_path.name} ...")
        d = torch.load(file_path, map_location="cpu", weights_only=False)
        
        if args.target_key not in d:
            print(f"Warning: Target key '{args.target_key}' not found in {file_path.name}. Skipping this file.")
            continue
            
        tgt = d[args.target_key].float()  # (B, wl_len, 2)
        
        # Simple sanity filter (absorptance must be within physical ranges)
        # Check max/min across all wavelengths and polarizations
        valid = (tgt.max(dim=2).values.max(dim=1).values <= 1.05) & (tgt.min(dim=2).values.min(dim=1).values >= -0.05)
        
        all_targets.append(tgt[valid])
        all_hs.append(d["h"].float()[valid])
        if "params_x" in d:
            all_params_x.append(d["params_x"].float()[valid])
            
    if not all_targets:
        print("No valid samples loaded.")
        return
        
    targets = torch.cat(all_targets) # (N, wl_len, 2)
    hs = torch.cat(all_hs).numpy()
    
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
    print(f"\n--- Top 5 Performing Structures ({args.target_key}) ---")
    print(f"{'Rank':<6}{'Avg Abs':<12}{'Height (nm)':<14}{'Max Amp (nm)':<16}")
    for rank, idx in enumerate(top_indices):
        h_val = hs[idx]
        abs_val = spectrum_averaged_abs[idx]
        
        max_amp = np.max(params_x[idx]) if params_x is not None else 0.0
        
        print(f"{rank+1:<6}{abs_val:<12.4f}{h_val:<14.2f}{max_amp:<16.2f}")
        
    # Setup output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = data_path / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left: Histogram of spectrum-averaged absorptances
    counts, bins, patches = ax1.hist(spectrum_averaged_abs, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax1.set_title(f"Spectrum-Averaged Absorptance\n({material_name})")
    ax1.set_xlabel(f"Average Absorptance ({args.target_key})")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Right: Scatter Plot of Wavelength vs Absorptance colored by wavelength
    # We tile wavelengths for all samples, and flatten the absorptance spectrum
    wls_scatter = np.tile(WAVELENGTHS, N)
    abs_scatter = avg_abs_spectrum.flatten()
    
    # Plot with small markers and alpha to handle the high density beautifully
    sc = ax2.scatter(wls_scatter, abs_scatter, c=wls_scatter, cmap='turbo', s=1, alpha=0.15, edgecolors='none', rasterized=True)
    
    # Add a clean colorbar
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Wavelength (nm)")
    
    ax2.set_title(f"Spectral Response Density\n({material_name})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel(f"Absorptance ({args.target_key})")
    ax2.set_xlim(280, 1120)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    plot_name = f"dataset_baseline_2d_{args.target_key}_{args.split}.png"
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
