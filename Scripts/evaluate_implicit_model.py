#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import MATERIAL_LIBRARY, N_MATERIALS, SIREN, build_profile
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.checkpoint import load_forward_model

MATERIAL_COLORS = {
    "Si":    "#1f77b4",
    "TiO2":  "#ff7f0e",
    "Si3N4": "#2ca02c",
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SIREN on a fine wavelength grid.")
    p.add_argument("--ckpt_dir", default="Checkpoints/Si_TiO2_Si3N4", help="Path to checkpoint dir")
    p.add_argument("--spacing", type=float, default=1.0, help="Wavelength spacing in nm (default: 1.0)")
    p.add_argument("--num_samples", type=int, default=5, help="Number of random samples to evaluate (default: 5)")
    p.add_argument("--use_val_data", action="store_true", help="Sample from the consolidated validation dataset instead of uniform random.")
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
    
    # Hardcode inc_ang limit to zero to evaluate normal incidence only
    geo_min[-1] = 0.0
    geo_max[-1] = 0.0
    
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
    
    if args.use_val_data:
        dataset_geos = []
        dataset_mat_ids = []
        dataset_targets = []
        for mat_name in trained_mats:
            val_path = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_name}" / "val_dataset.pt"
            if val_path.exists():
                data = torch.load(val_path, map_location="cpu", weights_only=False)
                px = data["params_x"].float()
                h = data["h"].float().unsqueeze(-1)
                inc = data["inc_ang"].float().unsqueeze(-1)
                geo = torch.cat([px.view(px.shape[0], -1), h, inc], dim=-1)
                
                dataset_geos.append(geo)
                dataset_targets.append(data["A_film_normal"].float())
                mat_id = list(MATERIAL_LIBRARY.keys()).index(mat_name)
                dataset_mat_ids.append(torch.full((geo.shape[0],), mat_id, dtype=torch.long))
        
        if dataset_geos:
            all_geos = torch.cat(dataset_geos).to(device)
            all_mats = torch.cat(dataset_mat_ids).to(device)
            all_targets = torch.cat(dataset_targets).to(device)
            indices = torch.randperm(len(all_geos))[:args.num_samples]
            test_geos = all_geos[indices]
            test_mat_ids = all_mats[indices]
            test_targets = all_targets[indices]
            
            # Hardcode inc_ang to 0.0 for normal incidence evaluation only
            test_geos[:, -1] = 0.0
        else:
            print("Warning: Could not find val_dataset.pt files. Falling back to uniform random sampling.")
            args.use_val_data = False
            test_targets = None
            
    if not args.use_val_data:
        test_geos = torch.rand(args.num_samples, len(geo_min), device=device) * (geo_max - geo_min) + geo_min
        test_mats = torch.randint(0, len(trained_mats), (args.num_samples,), device=device)
        # Map to global library indices
        test_mat_ids = torch.tensor([list(MATERIAL_LIBRARY.keys()).index(trained_mats[m]) for m in test_mats], device=device)
        test_targets = None
    
    print("Running evaluations...")
    for i in tqdm(range(args.num_samples), desc="Simulating"):
        geo = test_geos[i:i+1]
        mat_id = test_mat_ids[i:i+1]
        mat_name = list(MATERIAL_LIBRARY.keys())[mat_id.item()]
        
        # 1. SIREN prediction
        t0 = time.time()
        with torch.no_grad():
            siren_pred = model(geo, mat_id, wls=fine_wls_tensor)
            siren_p = siren_pred[0, :, 0].cpu().numpy()
            siren_s = siren_pred[0, :, 1].cpu().numpy()
        siren_time = time.time() - t0
            
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
        
        t0 = time.time()
        A_film, _ = get_absorptance_curve(
            params_x=px, params_y=None, wavelengths=torcwa_wls_tensor.cpu(), config=base_config, show_progress=True
        )
        torcwa_time = time.time() - t0
        rcwa_p = A_film[:, 0].cpu().numpy()
        rcwa_s = A_film[:, 1].cpu().numpy()
        
        # 3. Torcwa Simulation on original dataset resolution
        orig_wls = np.linspace(300, 1100, 161)
        if test_targets is not None:
            rcwa_p_orig = test_targets[i, :, 0].cpu().numpy()
            rcwa_s_orig = test_targets[i, :, 1].cpu().numpy()
        else:
            orig_wls_tensor = torch.tensor(orig_wls, dtype=torch.float64, device=device)
            A_film_orig, _ = get_absorptance_curve(
                params_x=px, params_y=None, wavelengths=orig_wls_tensor.cpu(), config=base_config, show_progress=False
            )
            rcwa_p_orig = A_film_orig[:, 0].cpu().numpy()
            rcwa_s_orig = A_film_orig[:, 1].cpu().numpy()
        
        # 4. Extract Geometry profile
        prof_tensor, _, _ = build_profile(geo.cpu(), n_harmonics, nx=128)
        profile_np = prof_tensor[0].numpy()
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), layout="constrained")
        
        cmap = plt.cm.viridis
        c_amp = cmap(0.3)
        c_phase = cmap(0.9)
        mat_color = MATERIAL_COLORS.get(mat_name, "steelblue")
        
        # P-Polarization
        ax_p = axes[0, 0]
        ax_p.plot(fine_wls, rcwa_p, "k-", lw=2, label="Torcwa (Ground Truth)")
        ax_p.plot(fine_wls, siren_p, "r--", lw=2, label="SIREN (Continuous)")
        if test_targets is not None:
            ax_p.scatter(orig_wls, rcwa_p_orig, color="k", edgecolor="k", s=30, alpha=0.8, zorder=5, label="Torcwa (161 pts)")
        ax_p.set_title("P-Polarization")
        ax_p.set_xlabel("Wavelength (nm)")
        ax_p.set_ylabel("Absorptance")
        ax_p.set_ylim(-0.05, 1.05)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend()
        
        # S-Polarization
        ax_s = axes[0, 1]
        ax_s.plot(fine_wls, rcwa_s, "k-", lw=2, label="Torcwa (Ground Truth)")
        ax_s.plot(fine_wls, siren_s, "r--", lw=2, label="SIREN (Continuous)")
        if test_targets is not None:
            ax_s.scatter(orig_wls, rcwa_s_orig, color="k", edgecolor="k", s=30, alpha=0.8, zorder=5, label="Torcwa (161 pts)")
        ax_s.set_title("S-Polarization")
        ax_s.set_xlabel("Wavelength (nm)")
        ax_s.set_ylim(-0.05, 1.05)
        ax_s.grid(True, alpha=0.3)
        
        # Structure Cross-Section
        ax_struct = axes[1, 0]
        xs = np.linspace(0, rcwa_config_dict.get("grating_period", 1000), 128)
        ax_struct.plot(xs, profile_np, "k-", lw=2)
        ax_struct.fill_between(xs, 0, profile_np, color=mat_color, alpha=0.6)
        ax_struct.set_title("Structure Cross-Section")
        ax_struct.set_xlabel("x (nm)")
        ax_struct.set_ylabel("Height (nm)")
        
        # Harmonics amplitudes & phases
        ax_h = axes[1, 1]
        px_np = px.numpy()
        amps_geo = px_np[:, 0]
        phases_geo = px_np[:, 1]
        x_pos = np.arange(1, n_harmonics + 1)
        ax_h.bar(x_pos, amps_geo, color=c_amp, edgecolor="black")
        ax_h.set_ylabel("Amplitude (nm)", color=c_amp)
        ax_h.tick_params(axis='y', labelcolor=c_amp, labelsize=12)
        ax_h.tick_params(axis='x', labelsize=12)
        ax_h.set_xlabel("Harmonic index")
        ax_h.set_title("Harmonic Composition")
        
        ax_p2 = ax_h.twinx()
        ax_p2.plot(x_pos, phases_geo, 'o', color=c_phase, markersize=10, markeredgecolor="black")
        ax_p2.set_ylabel("Phase (rad)", color=c_phase)
        ax_p2.tick_params(axis='y', labelcolor=c_phase, labelsize=12)
        ax_p2.set_ylim(-0.5, 2 * np.pi + 0.5)
        
        # Calculate MAE
        mae_fine = (np.abs(siren_p - rcwa_p).mean() + np.abs(siren_s - rcwa_s).mean()) / 2.0
        mae_fine_str = f" | MAE (801pt): {mae_fine:.4f}"
        
        if test_targets is not None:
            with torch.no_grad():
                orig_wls_tensor = torch.from_numpy(orig_wls).float().to(device)
                siren_coarse = model(geo, mat_id, wls=orig_wls_tensor)
                s_c_p = siren_coarse[0, :, 0].cpu().numpy()
                s_c_s = siren_coarse[0, :, 1].cpu().numpy()
            mae_coarse = (np.abs(s_c_p - rcwa_p_orig).mean() + np.abs(s_c_s - rcwa_s_orig).mean()) / 2.0
            mae_coarse_str = f" | MAE (161pt): {mae_coarse:.4f}"
        else:
            mae_coarse_str = ""
            
        speedup = torcwa_time / siren_time if siren_time > 0 else 0
        plt.suptitle(f"SIREN Implicit Generalization: Sample {i+1} ({mat_name})\n$h={h_val:.1f}$ nm, $\\theta={inc_ang:.1f}^\\circ${mae_coarse_str}{mae_fine_str}\nSIREN: {siren_time*1000:.1f}ms | Torcwa: {torcwa_time:.1f}s | Speedup: {speedup:.0f}x", fontsize=16)
        plt.savefig(out_dir / f"siren_fine_grid_sample_{i+1}.png", dpi=300)
        plt.close(fig)

    print(f"Finished! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
