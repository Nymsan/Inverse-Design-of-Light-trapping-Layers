#!/usr/bin/env python
"""
Evaluate trained inverse models and generate performance report.

Usage:
    python Scripts/evaluate_inverse.py --ckpt_dir Checkpoints/Si_TiO2_Si3N4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import json
from pathlib import Path

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import (
    MATERIAL_LIBRARY, N_MATERIALS,
    ForwardMLP, SpatialCNN, SkipCNN, SIREN, TransformerForward,
    TandemNetwork, GenerativeTandemNetwork, ContrastiveVAE, InverseDecoder,
    GeometryEncoder, SpectrumEncoder, GeometryDecoder
)
from Utils.utils import generate_test_batch, get_absorptance_curve, RCWAConfig
from Scripts.train_inverse import get_best_forward_model

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

WAVELENGTHS = np.linspace(300, 1100, 161)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    return p.parse_args()


@torch.no_grad()
def plot_model_dashboard(
    model_name: str,
    inv_model: nn.Module,
    forward_model: nn.Module,
    targets: torch.Tensor,
    is_ideal: list[bool],
    rcwa_config_dict: dict,
    stats: dict,
    save_path: str,
    n_wavelengths: int
):
    """
    Creates a comprehensive dashboard for a single inverse model.
    Rows: Each target.
    Cols:
      1. p-pol Spectrum (Target vs Surrogate vs Torcwa)
      2. s-pol Spectrum (Target vs Surrogate vs Torcwa)
      3. Grating Profile
      4. Harmonic Amplitudes & Phases
    """
    device = next(inv_model.parameters()).device
    n_samples = targets.shape[0]
    n_wl_half = n_wavelengths // 2
    
    # Generate designs
    curve = targets.to(device)
    if hasattr(inv_model, "sample_diverse_designs"):
        designs = inv_model(curve, z=torch.randn(n_samples, inv_model.latent_dim, device=device))
        pred_geo = designs["pred_geometry"]
        mat_oh = designs["material_onehot"]
    elif hasattr(inv_model, "spectrum_encoder"):
        z_y = inv_model.spectrum_encoder(curve)
        z_noisy = z_y + torch.randn_like(z_y) * 0.5
        pred_geo, mat_oh, _ = inv_model.geometry_decoder(z_noisy, tau=0.1, hard=True)
    elif hasattr(inv_model, "inverse_decoder"):
        pred_geo, mat_oh, _ = inv_model.inverse_decoder(curve, tau=0.1, hard=True)
    else:
        pred_geo, mat_oh, _ = inv_model(curve)
        
    # Forward Surrogate Prediction
    surrogate_preds = forward_model(pred_geo, mat_oh.argmax(dim=-1))
    
    # Setting up the figure
    fig, axes = plt.subplots(n_samples, 4, figsize=(22, 5 * n_samples), squeeze=False, layout="tight")
    mat_names = list(stats["materials"].keys())
    
    n_harmonics = stats["n_harmonics"]
    n_fourier = n_harmonics * 2
    r_grid = np.linspace(0, 1000.0, 256)
    harmonic_idx = np.arange(1, n_harmonics + 1)
    
    for i in tqdm(range(n_samples), desc=f"Evaluating {model_name} (Torcwa Simulation)"):
        mse_list = []
        # Geometry extraction
        px = pred_geo[i, :n_fourier].view(-1, 2)
        h_nm = pred_geo[i, n_fourier].item()
        inc_ang_deg = pred_geo[i, n_fourier+1].item() if pred_geo.shape[1] > n_fourier+1 else 0.0
        
        pred_mat_idx = mat_oh[i].argmax().item()
        pred_mat_name = mat_names[pred_mat_idx] if pred_mat_idx < len(mat_names) else "Si"
        
        # 1. Torcwa Simulation
        base_config = RCWAConfig(**rcwa_config_dict)
        base_config.h = float(h_nm)
        base_config.inc_ang = (float(inc_ang_deg) + 1e-3) * np.pi/180
        base_config.azi_ang = 1e-3 * np.pi/180
        if pred_mat_name.endswith("_Ag"):
            base_config.grating_material = pred_mat_name[:-3]
            base_config.reflector_type = 'Ag'
        else:
            base_config.grating_material = pred_mat_name
            base_config.reflector_type = 'pec'
        
        A_film, _ = get_absorptance_curve(
            params_x=px,
            params_y=None,
            wavelengths=torch.from_numpy(WAVELENGTHS).double(),
            config=base_config,
            show_progress=False
        )
        rcwa_p = A_film[:, 0].cpu().numpy()
        rcwa_s = A_film[:, 1].cpu().numpy()
        
        target_np = curve[i].cpu().numpy()
        sim_np = np.concatenate([rcwa_p, rcwa_s])
        mse_list.append(np.mean((target_np - sim_np)**2))
        
        # 2. P-Pol Spectra
        ax_p = axes[i, 0]
        ax_p.plot(WAVELENGTHS, curve[i, :n_wl_half].cpu().numpy(), "k-", lw=2.5, label="Target", zorder=10)
        ax_p.plot(WAVELENGTHS, surrogate_preds[i, :n_wl_half].cpu().numpy(), "r--", lw=2.0, label="Surrogate")
        ax_p.plot(WAVELENGTHS, rcwa_p, "g-", lw=2.0, label="Torcwa Physics")
        ax_p.set_xlim(300, 1100); ax_p.set_ylim(-0.05, 1.05)
        ax_p.set_ylabel("Absorptance (P-Pol)")
        if i == 0:
            ax_p.legend(fontsize=9)
        title_prefix = "Ideal Target" if is_ideal[i] else "Real Target"
        ax_p.set_title(f"{title_prefix} P-Pol")
        
        # 3. S-Pol Spectra
        ax_s = axes[i, 1]
        ax_s.plot(WAVELENGTHS, curve[i, n_wl_half:].cpu().numpy(), "k-", lw=2.5, label="Target", zorder=10)
        ax_s.plot(WAVELENGTHS, surrogate_preds[i, n_wl_half:].cpu().numpy(), "b--", lw=2.0, label="Surrogate")
        ax_s.plot(WAVELENGTHS, rcwa_s, "g-", lw=2.0, label="Torcwa Physics")
        ax_s.set_xlim(300, 1100); ax_s.set_ylim(-0.05, 1.05)
        ax_s.set_ylabel("Absorptance (S-Pol)")
        ax_s.set_title(f"{title_prefix} S-Pol")
        
        # 4. Grating Profile
        ax_g = axes[i, 2]
        amps = px[:, 0].cpu().numpy()
        phases = px[:, 1].cpu().numpy()
        
        grating_height = 2.0 * amps.sum() + 1e-9
        arg = 2.0 * np.pi * harmonic_idx[:, None] * r_grid[None, :] / 1000.0 - phases[:, None]
        cosines = amps[:, None] * np.cos(arg)
        prof = grating_height / 2.0 + cosines.sum(axis=0)
        
        ax_g.fill_between(r_grid, 0, prof, color="gray", alpha=0.5)
        ax_g.set_ylim(0, max(120, grating_height * 1.2))
        ax_g.set_xlim(0, 1000)
        ax_g.set_title(f"Predicted Profile ({pred_mat_name})")
        ax_g.set_ylabel("Thickness (nm)")
        
        text_str = f"Material: {pred_mat_name}\nFilm Height ($h$): {h_nm:.1f} nm\nGrating Height: {grating_height:.1f} nm\nIncidence: {inc_ang_deg:.1f}°"
        ax_g.text(0.05, 0.95, text_str, transform=ax_g.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        # 5. Harmonics Bar Plot
        ax_h = axes[i, 3]
        x_pos = np.arange(1, n_harmonics + 1)
        ax_h.bar(x_pos, amps, color="skyblue", edgecolor="black")
        ax_h.set_ylabel("Amplitude (nm)", color="C0")
        ax_h.tick_params(axis='y', labelcolor="C0")
        
        ax_p2 = ax_h.twinx()
        ax_p2.plot(x_pos, phases, 'ro', markersize=8)
        ax_p2.set_ylabel("Phase (rad)", color="red")
        ax_p2.tick_params(axis='y', labelcolor="red")
        ax_p2.set_ylim(-0.5, 2*np.pi + 0.5)
        
        ax_h.set_title("Harmonics Amplitudes & Phases")
        ax_h.set_xticks(x_pos)
        ax_h.set_xticklabels([f"n={n}" for n in x_pos])
        
        if i == n_samples - 1:
            ax_p.set_xlabel("Wavelength (nm)")
            ax_s.set_xlabel("Wavelength (nm)")
            ax_g.set_xlabel("x (nm)")
            ax_h.set_xlabel("Harmonic Index")
            
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved Dashboard: {save_path}")
    return np.mean(mse_list)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "evaluation" / "inverse"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    mat_names = list(stats["materials"].keys())
    print(f"Loaded Stats: n_continuous={n_continuous}, n_wavelengths={n_wavelengths}, materials={mat_names}")

    batch = generate_test_batch(stats)
    
    forward_model, fwd_name, fwd_loss = get_best_forward_model(ckpt_dir, n_continuous, n_wavelengths, n_harmonics)
    if forward_model is not None:
        forward_model.eval()
        print(f"\n=> Loaded BEST forward model for evaluation: {fwd_name}")
    else:
        print("\n=> ERROR: No forward models found.")
        return

    def safe_load_state_dict(model, state_dict):
        current_state = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in current_state:
                curr_v = current_state[k]
                if v.shape != curr_v.shape:
                    if "material_embedding" in k or "material_head" in k:
                        if v.shape[0] < curr_v.shape[0]:
                            new_w = curr_v.clone()
                            new_w[:v.shape[0]] = v
                            state_dict[k] = new_w
        model.load_state_dict(state_dict, strict=False)

    inverse_models = {}
    
    # Load Models
    tandem_path = ckpt_dir / "tandem.pt"
    if tandem_path.exists():
        dec = InverseDecoder(n_wavelengths=n_wavelengths, n_geometry=n_continuous, n_materials=N_MATERIALS, latent_dim=0, geo_min=stats["geo_min"], geo_max=stats["geo_max"])
        m = TandemNetwork(inverse_decoder=dec, forward_model=forward_model)
        safe_load_state_dict(m, torch.load(tandem_path, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        inverse_models["tandem"] = m

    gen_path = ckpt_dir / "generative_tandem.pt"
    if gen_path.exists():
        dec = InverseDecoder(n_wavelengths=n_wavelengths, n_geometry=n_continuous, n_materials=N_MATERIALS, latent_dim=32, geo_min=stats["geo_min"], geo_max=stats["geo_max"])
        m = GenerativeTandemNetwork(inverse_decoder=dec, forward_model=forward_model, latent_dim=32)
        safe_load_state_dict(m, torch.load(gen_path, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        inverse_models["generative_tandem"] = m
        
    cvae_path = ckpt_dir / "cvae.pt"
    if cvae_path.exists():
        g_e = GeometryEncoder(n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8, latent_dim=64, fc_dims=(256, 256))
        g_d = GeometryDecoder(latent_dim=64, n_geometry=n_continuous, n_materials=N_MATERIALS, hidden_dims=(256, 256), geo_min=stats["geo_min"], geo_max=stats["geo_max"])
        s_e = SpectrumEncoder(n_wavelengths=n_wavelengths, latent_dim=64, conv_channels=(32, 64, 128, 64), fc_dims=(256, 256))
        m = ContrastiveVAE(geometry_encoder=g_e, geometry_decoder=g_d, spectrum_encoder=s_e, margin_radius=1.0, beta=1e-3, gamma=1.0)
        safe_load_state_dict(m, torch.load(cvae_path, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        inverse_models["cvae"] = m

    cvae_wishful_path = ckpt_dir / "cvae_wishful.pt"
    if cvae_wishful_path.exists():
        g_e = GeometryEncoder(n_continuous=n_continuous, n_materials=N_MATERIALS, embed_dim=8, latent_dim=64, fc_dims=(256, 256))
        g_d = GeometryDecoder(latent_dim=64, n_geometry=n_continuous, n_materials=N_MATERIALS, hidden_dims=(256, 256), geo_min=stats["geo_min"], geo_max=stats["geo_max"])
        s_e = SpectrumEncoder(n_wavelengths=n_wavelengths, latent_dim=64, conv_channels=(32, 64, 128, 64), fc_dims=(256, 256))
        m = ContrastiveVAE(geometry_encoder=g_e, geometry_decoder=g_d, spectrum_encoder=s_e, margin_radius=1.0, beta=1e-3, gamma=1.0)
        safe_load_state_dict(m, torch.load(cvae_wishful_path, map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        inverse_models["cvae_wishful"] = m

    # Retrieve RCWA Config
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_names[0]}" / "batch_0000.pt"
    if first_batch_file.exists():
        rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    else:
        rcwa_config_dict = {}

    print("\nGenerating comprehensive model dashboards...")
    
    # We will pick 2 real targets from the batch, and 2 ideal broadband targets.
    real_targets = batch["target"][:2]
    ideal_target = torch.ones(2, n_wavelengths)
    
    targets_to_eval = torch.cat([real_targets, ideal_target], dim=0)
    is_ideal = [False, False, True, True]
    
    all_metrics = {}

    for name, m in inverse_models.items():
        save_path = str(eval_dir / f"dashboard_{name}.png")
        avg_mse = plot_model_dashboard(
            model_name=name,
            inv_model=m,
            forward_model=forward_model,
            targets=targets_to_eval,
            is_ideal=is_ideal,
            rcwa_config_dict=rcwa_config_dict,
            stats=stats,
            save_path=save_path,
            n_wavelengths=n_wavelengths
        )
        all_metrics[name] = {"mse": float(avg_mse)}

    metrics_path = eval_dir / "inverse_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll evaluation outputs and metrics saved to: {eval_dir}")

if __name__ == "__main__":
    main()
