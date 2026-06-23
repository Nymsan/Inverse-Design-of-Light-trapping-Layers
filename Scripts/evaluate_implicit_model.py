#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import MATERIAL_LIBRARY, N_MATERIALS, SIREN
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.checkpoint import load_forward_model

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SIREN on a fine wavelength grid.")
    p.add_argument("--ckpt_dir", default="Checkpoints/Si_TiO2_Si3N4", help="Path to checkpoint dir")
    p.add_argument("--spacing", type=float, default=1.0, help="Wavelength spacing in nm (default: 1.0)")
    p.add_argument("--num_samples", type=int, default=5, help="Number of random samples to evaluate (default: 5)")
    return p.parse_args()

def main():
    args = parse_args()
    provided_path = Path(args.ckpt_dir)
    if provided_path.is_absolute() or provided_path.exists():
        ckpt_dir = provided_path.resolve()
    else:
        ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        print(f"Error: {stats_path} not found.")
        return
        
    stats = torch.load(stats_path, map_location="cpu")
    geo_min = stats["geo_min"].to(device)
    geo_max = stats["geo_max"].to(device)
    n_harmonics = stats["n_harmonics"]
    n_fourier = n_harmonics * 2
    trained_mats = stats["materials"]
    
    # Load SIREN model specifically
    siren_ckpt = ckpt_dir / "siren.pt"
    if not siren_ckpt.exists():
        print(f"Error: SIREN checkpoint not found at {siren_ckpt}")
        return
        
    model, _, _ = load_forward_model(
        siren_ckpt, 
        n_continuous=stats["n_continuous"], 
        n_wavelengths=stats["n_wavelengths"], 
        n_harmonics=n_harmonics
    )
    model.to(device)
    model.eval()
    
    fine_wls = np.arange(300, 1100 + args.spacing, args.spacing)
    fine_wls_tensor = torch.tensor(fine_wls, dtype=torch.float32, device=device)
    torcwa_wls_tensor = torch.tensor(fine_wls, dtype=torch.float64, device=device)
    
    print(f"Evaluating SIREN on fine grid: 300 to 1100 nm (spacing = {args.spacing} nm) | {len(fine_wls)} points")
    
    rcwa_config_dict = {
        'grating_period': 1000.0,
        'order_N': 15,
        'nx': 128,
        'height_per_layer': 5.0,
    }
    
    out_dir = ckpt_dir / "evaluation" / "implicit_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate random test geometries
    torch.manual_seed(42)
    test_geos = torch.rand(args.num_samples, len(geo_min), device=device) * (geo_max - geo_min) + geo_min
    test_mats = torch.randint(0, len(trained_mats), (args.num_samples,), device=device)
    # Map to global library indices
    test_mat_ids = torch.tensor([list(MATERIAL_LIBRARY.keys()).index(trained_mats[m]) for m in test_mats], device=device)
    
    print("Running evaluations...")
    for i in tqdm(range(args.num_samples), desc="Simulating"):
        geo = test_geos[i:i+1]
        mat_id = test_mat_ids[i:i+1]
        mat_name = list(MATERIAL_LIBRARY.keys())[mat_id.item()]
        
        # 1. SIREN prediction
        with torch.no_grad():
            siren_pred = model(geo, mat_id, wls=fine_wls_tensor)
            siren_p = siren_pred[0, :, 0].cpu().numpy()
            siren_s = siren_pred[0, :, 1].cpu().numpy()
            
        # 2. Torcwa Simulation
        base_config = RCWAConfig(**rcwa_config_dict)
        h_val = geo[0, -2].item()
        inc_ang = geo[0, -1].item()
        
        base_config.h = float(h_val)
        base_config.inc_ang = (float(inc_ang) + 1e-3) * np.pi/180.0
        base_config.azi_ang = 1e-3 * np.pi/180.0
        
        if mat_name.endswith("_Ag"):
            base_config.grating_material = mat_name[:-3]
            base_config.reflector_type = 'Ag'
        else:
            base_config.grating_material = mat_name
            base_config.reflector_type = 'pec'
            
        px = geo[0, :n_fourier].view(-1, 2).to(torch.float32).cpu()
        
        A_film, _ = get_absorptance_curve(
            params_x=px, params_y=None, wavelengths=torcwa_wls_tensor.cpu(), config=base_config, show_progress=False
        )
        rcwa_p = A_film[:, 0].cpu().numpy()
        rcwa_s = A_film[:, 1].cpu().numpy()
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        
        axes[0].plot(fine_wls, rcwa_p, "k-", lw=2, label="Torcwa (Ground Truth)")
        axes[0].plot(fine_wls, siren_p, "r--", lw=2, label="SIREN (Continuous)")
        axes[0].set_title("P-Polarization")
        axes[0].set_xlabel("Wavelength (nm)")
        axes[0].set_ylabel("Absorptance")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(fine_wls, rcwa_s, "k-", lw=2, label="Torcwa (Ground Truth)")
        axes[1].plot(fine_wls, siren_s, "r--", lw=2, label="SIREN (Continuous)")
        axes[1].set_title("S-Polarization")
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"SIREN Implicit Generalization: Sample {i+1} ({mat_name})\n$h={h_val:.1f}$ nm, $\\theta={inc_ang:.1f}^\\circ$", fontsize=14)
        plt.savefig(out_dir / f"siren_fine_grid_sample_{i+1}.png")
        plt.close(fig)

    print(f"Finished! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
