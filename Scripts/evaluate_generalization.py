#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import MATERIAL_LIBRARY, N_MATERIALS, build_profile
from Utils.utils import get_absorptance_curve, RCWAConfig
from Utils.checkpoint import load_forward_model, _FORWARD_FILENAMES

def get_lhs_samples(num_samples, geo_min, geo_max, n_harmonics=20, seed=42):
    geo_min_np = geo_min.cpu().numpy()
    geo_max_np = geo_max.cpu().numpy()
    
    n_harmonics_train = (len(geo_max_np) - 2) // 2
    
    h_min = geo_min_np[-2]
    h_max = geo_max_np[-2]
    inc_min = geo_min_np[-1]
    inc_max = geo_max_np[-1]
    
    amp_max_train = np.max(geo_max_np[0:2*n_harmonics_train:2])
    
    sampler = qmc.LatinHypercube(d=2 + n_harmonics * 2, seed=seed)
    sample = sampler.random(n=num_samples)
    
    l_bounds = [h_min, inc_min] + [0.0] * n_harmonics + [0.0] * n_harmonics
    u_bounds = [h_max, inc_max] + [amp_max_train] * n_harmonics + [2*np.pi] * n_harmonics
    
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    h = scaled_sample[:, 0]
    inc_ang = scaled_sample[:, 1]
    amps = scaled_sample[:, 2:2+n_harmonics]
    phases = scaled_sample[:, 2+n_harmonics:]
    return h, inc_ang, amps, phases

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate forward models on generalized high-harmonic dataset")
    p.add_argument("--ckpt_dir", default="Checkpoints/Si_TiO2_Si3N4", help="Path to checkpoint dir")
    p.add_argument("--num_samples", type=int, default=5, help="Number of high-harmonic samples to simulate")
    p.add_argument("--n_harmonics", type=int, default=20, help="Number of harmonics to evaluate generalization on")
    return p.parse_args()

def main():
    args = parse_args()
    provided_path = Path(args.ckpt_dir)
    if provided_path.is_absolute() or provided_path.exists():
        ckpt_dir = provided_path.resolve()
    else:
        ckpt_dir = PROJECT_ROOT / args.ckpt_dir
        
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        print(f"Stats not found at {stats_path}")
        return
        
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    trained_mat_names = stats["materials"]
    
    out_dir = ckpt_dir / "evaluation" / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_harmonics_test = args.n_harmonics
    
    print(f"Generating {args.num_samples} LHS samples with {n_harmonics_test} harmonics...")
    h, inc_ang, amps, phases = get_lhs_samples(args.num_samples, stats["geo_min"], stats["geo_max"], n_harmonics=n_harmonics_test)
    
    WAVELENGTHS = np.linspace(300, 1100, stats["n_wavelengths"] // 2)
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{trained_mat_names[0]}" / "batch_0000.pt"
    if first_batch_file.exists():
        rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    else:
        rcwa_config_dict = {}
    
    all_physics_targets = []
    sampled_materials = []
    
    px_tensor = torch.from_numpy(np.stack([amps, phases], axis=-1)).float()
    h_tensor = torch.from_numpy(h).float().unsqueeze(1)
    inc_tensor = torch.from_numpy(inc_ang).float().unsqueeze(1)
    
    for i in tqdm(range(args.num_samples), desc="Simulating Torcwa Physics"):
        mat_name = trained_mat_names[i % len(trained_mat_names)]
        sampled_materials.append(mat_name)
        
        base_config = RCWAConfig(**rcwa_config_dict)
        base_config.h = float(h[i])
        base_config.inc_ang = (float(inc_ang[i]) + 1e-3) * np.pi/180
        base_config.azi_ang = 1e-3 * np.pi/180
        
        if mat_name.endswith("_Ag"):
            base_config.grating_material = mat_name[:-3]
            base_config.reflector_type = 'Ag'
        else:
            base_config.grating_material = mat_name
            base_config.reflector_type = 'pec'
            
        A_film, _ = get_absorptance_curve(
            params_x=px_tensor[i],
            params_y=None,
            wavelengths=torch.from_numpy(WAVELENGTHS).double(),
            config=base_config,
            show_progress=False
        )
        all_physics_targets.append(A_film.cpu())
        
    physics_targets = torch.stack(all_physics_targets, dim=0)
    
    # Run Forward Models
    mat_indices = torch.tensor([MATERIAL_LIBRARY[m] for m in sampled_materials], dtype=torch.long, device=device)
    
    # Run Forward Models by explicitly passing the generated profile
    # This prevents the models from relying on their fixed internal 'n_harmonics' weights 
    # and forces them to generalize over the raw spatial structure directly!
    mat_indices = torch.tensor([MATERIAL_LIBRARY[m] for m in sampled_materials], dtype=torch.long, device=device)
    
    true_geo = torch.cat([px_tensor.view(args.num_samples, -1), h_tensor, inc_tensor], dim=1).to(device)
    profile, h_model, inc_model = build_profile(true_geo, n_harmonics=args.n_harmonics, nx=128)
    
    samples_data = []
    for i in range(args.num_samples):
        samples_data.append({
            "sample_id": i + 1,
            "material": sampled_materials[i],
            "h": float(h[i]),
            "inc_ang": float(inc_ang[i]),
            "amps": amps[i].tolist(),
            "phases": phases[i].tolist()
        })
        
    metrics = {"samples": samples_data}
    
    for fname in _FORWARD_FILENAMES:
        ckpt_path = ckpt_dir / fname
        if not ckpt_path.exists():
            continue
            
        model, hist, name = load_forward_model(
            ckpt_path, n_continuous=stats["n_continuous"], n_wavelengths=stats["n_wavelengths"], n_harmonics=stats["n_harmonics"]
        )
        model = model.to(device)
        model.eval()
        
        try:
            with torch.no_grad():
                preds = model(geometry=None, material_id=mat_indices, profile=profile, h=h_model, inc_ang=inc_model).cpu()
        except Exception as e:
            print(f"Skipping {name} due to shape mismatch on {args.n_harmonics}-harmonic input (expected): {e}")
            continue
            
        target_flat = torch.cat([physics_targets[:, :, 0], physics_targets[:, :, 1]], dim=1)
        mae = torch.mean(torch.abs(preds - target_flat)).item()
        metrics[name] = {"mae": mae}
        
        # Plot dashboard
        cmap = plt.cm.viridis
        c_surr, c_physics = cmap(0.5), cmap(0.8)
        c_amp = cmap(0.3)
        c_phase = cmap(0.9)
        
        fig, axes = plt.subplots(args.num_samples, 3, figsize=(18, 5 * args.num_samples), squeeze=False)
        for i in range(args.num_samples):
            ax_p = axes[i, 0]
            ax_s = axes[i, 1]
            ax_h = axes[i, 2]
            n_wl = stats["n_wavelengths"] // 2
            
            h_val = float(h[i])
            inc_val = float(inc_ang[i])
            mat_name = sampled_materials[i]
            
            title_p = f"Sample {i+1} ({mat_name}) - P-Pol\nh: {h_val:.1f}nm, inc: {inc_val:.1f}°"
            title_s = f"Sample {i+1} ({mat_name}) - S-Pol\nh: {h_val:.1f}nm, inc: {inc_val:.1f}°"
            
            ax_p.plot(WAVELENGTHS, physics_targets[i, :, 0].numpy(), linestyle="-", color=c_physics, lw=2.5, label="Torcwa Physics")
            ax_p.plot(WAVELENGTHS, preds[i, :n_wl].numpy(), linestyle="-", color=c_surr, lw=2, label="Surrogate")
            ax_p.set_title(title_p, fontsize=14)
            ax_p.set_ylim(-0.05, 1.05)
            ax_p.tick_params(axis='both', which='major', labelsize=10)
            if i == 0: ax_p.legend(fontsize=10)
            
            ax_s.plot(WAVELENGTHS, physics_targets[i, :, 1].numpy(), linestyle="-", color=c_physics, lw=2.5, label="Torcwa Physics")
            ax_s.plot(WAVELENGTHS, preds[i, n_wl:].numpy(), linestyle="-", color=c_surr, lw=2, label="Surrogate")
            ax_s.set_title(title_s, fontsize=14)
            ax_s.set_ylim(-0.05, 1.05)
            ax_s.tick_params(axis='both', which='major', labelsize=10)
            
            # Harmonics
            amps_geo = amps[i]
            phases_geo = phases[i]
            n_harm = len(amps_geo)
            x_pos = np.arange(1, n_harm + 1)
            
            ax_h.bar(x_pos, amps_geo, color=c_amp, edgecolor="black")
            ax_h.set_ylabel("Amplitude (nm)", color=c_amp, fontsize=12)
            ax_h.tick_params(axis='y', labelcolor=c_amp, labelsize=10)
            ax_h.tick_params(axis='x', labelsize=10)
            ax_h.set_xlabel("Harmonic index", fontsize=12)
            ax_h.set_title("Harmonic Composition", fontsize=14)
            
            ax_p2 = ax_h.twinx()
            ax_p2.plot(x_pos, phases_geo, 'o', color=c_phase, markersize=6, markeredgecolor="black")
            ax_p2.set_ylabel("Phase (rad)", color=c_phase, fontsize=12)
            ax_p2.tick_params(axis='y', labelcolor=c_phase, labelsize=10)
            ax_p2.set_ylim(-0.5, 2 * np.pi + 0.5)
            
        plt.tight_layout()
        save_path = out_dir / f"dashboard_{name}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved Dashboard for {name}: {save_path}")
        
    with open(out_dir / "generalization_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nAll generalization metrics and dashboards saved to: {out_dir}")

if __name__ == "__main__":
    main()
