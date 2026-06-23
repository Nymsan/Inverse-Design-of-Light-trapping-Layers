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
from Utils.checkpoint import get_best_forward_model, load_inverse_model, load_forward_model
from Scripts.evaluate_dataset_baseline import get_dataset_baseline

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150,
})

WAVELENGTHS = np.linspace(300, 1100, 161)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    p.add_argument("--force_forward_model", type=str, default=None, help="Force load a specific forward model (e.g. 'skip_cnn.pt')")
    p.add_argument(
        "--bands", nargs="+", type=float, default=None,
        metavar="WL",
        help="Band pairs (nm) for the ideal target step function, e.g. --bands 500 750 850 1000"
    )
    p.add_argument(
        "--material", type=str, default=None,
        help="Ask stochastic inverse models (GenTandem/CVAE) to predict the given material "
             "for the band-target rows (resampled up to 16 times). Example: --material TiO2"
    )
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
    n_wavelengths: int,
    bands: list[tuple[float, float]] | None = None,
    best_dataset_curve: torch.Tensor | None = None,
    best_dataset_abs: float | None = None,
    best_dataset_mat: str | None = None,
    requested_material: str | None = None,
    forward_model_name: str = "",
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
    
    is_stochastic = hasattr(inv_model, "sample_diverse_designs") or hasattr(inv_model, "spectrum_encoder")
    req_mat_idx = MATERIAL_LIBRARY.get(requested_material, -1) if requested_material else -1
    N_TRIES = 16

    def _sample_one(single_curve: torch.Tensor):
        """Generate one design for a (1, n_wl) curve tensor."""
        if hasattr(inv_model, "sample_diverse_designs"):
            d = inv_model(single_curve, z=torch.randn(1, inv_model.latent_dim, device=device))
            return d["pred_geometry"], d["material_onehot"]
        elif hasattr(inv_model, "spectrum_encoder"):
            z_y = inv_model.spectrum_encoder(single_curve)
            z_noisy = z_y + torch.randn_like(z_y) * 0.5
            g, m, _ = inv_model.geometry_decoder(z_noisy, tau=0.1, hard=True)
            return g, m
        elif hasattr(inv_model, "inverse_decoder"):
            g, m, _ = inv_model.inverse_decoder(single_curve, tau=0.1, hard=True)
            return g, m
        else:
            g, m, _ = inv_model(single_curve)
            return g, m

    # Generate designs — one per target row
    pred_geo_list, mat_oh_list = [], []
    curve = targets.to(device)
    for i in range(n_samples):
        single_curve = curve[i:i+1]
        g, m = _sample_one(single_curve)
        # For ideal rows of stochastic models, resample if a specific material is requested
        if is_ideal[i] and is_stochastic and req_mat_idx >= 0:
            for _ in range(N_TRIES - 1):
                if m[0].argmax().item() == req_mat_idx:
                    break
                g, m = _sample_one(single_curve)
        pred_geo_list.append(g)
        mat_oh_list.append(m)
    pred_geo = torch.cat(pred_geo_list, dim=0)
    mat_oh = torch.cat(mat_oh_list, dim=0)
        
    # Forward Surrogate Prediction
    surrogate_preds = forward_model(pred_geo, mat_oh.argmax(dim=-1))
    
    # Setting up the figure
    fig, axes = plt.subplots(n_samples, 4, figsize=(24, 6 * n_samples), squeeze=False, layout="constrained")
    mat_names = stats["materials"]
    
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
        
        # Use viridis colormap for consistency
        cmap = plt.cm.viridis
        c_surr, c_physics = cmap(0.45), cmap(0.85)
        c_profile, c_amp, c_phase = cmap(0.65), cmap(0.3), cmap(0.9)
        c_dataset = cmap(0.15)   # dark indigo — distinct from target (black), surrogate, torcwa

        
        # Compute in-band average absorptance for Torcwa result
        if bands:
            band_mask = np.zeros(len(WAVELENGTHS), dtype=bool)
            for bmin, bmax in bands:
                band_mask |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
            rcwa_avg_abs = float((rcwa_p[band_mask].mean() + rcwa_s[band_mask].mean()) / 2.0)
        else:
            rcwa_avg_abs = float((rcwa_p.mean() + rcwa_s.mean()) / 2.0)

        # Surrogate in-band absorptance for title
        if bands:
            surr_avg_abs = float(
                (surrogate_preds[i, :n_wl_half].cpu().numpy()[band_mask].mean()
                 + surrogate_preds[i, n_wl_half:].cpu().numpy()[band_mask].mean()) / 2.0
            )
        else:
            surr_avg_abs = float(surrogate_preds[i].mean().item())

        # 2. P-Pol Spectra
        ax_p = axes[i, 0]
        ax_p.plot(WAVELENGTHS, curve[i, :n_wl_half].cpu().numpy(), "k-", lw=2.5, label="Target", zorder=10)
        if best_dataset_curve is not None and is_ideal[i]:
            ds_label = f"Best Dataset ({best_dataset_mat})"
            ax_p.plot(WAVELENGTHS, best_dataset_curve[:n_wl_half].numpy(), color=c_dataset,
                      linestyle="-", lw=1.8, label=ds_label, alpha=0.75, zorder=9)
        ax_p.plot(WAVELENGTHS, surrogate_preds[i, :n_wl_half].cpu().numpy(), linestyle="--", color=c_surr, lw=2.0, label="Surrogate")
        ax_p.plot(WAVELENGTHS, rcwa_p, linestyle="-", color=c_physics, lw=2.0, label="Torcwa")
        ax_p.set_xlim(300, 1100); ax_p.set_ylim(-0.05, 1.05)
        ax_p.set_xlabel("Wavelength (nm) — P-Pol")
        ax_p.set_ylabel("Absorptance")
        if bands:
            for bmin, bmax in bands:
                ax_p.axvspan(bmin, bmax, color="gray", alpha=0.2)
        if i == 0 or is_ideal[i]:
            ax_p.legend(fontsize=9)
        if bands is not None:
            bands_str = ", ".join([f"{b[0]}-{b[1]}nm" for b in bands])
        else:
            bands_str = "Custom"
        title_prefix = f"Band Target ({bands_str})" if is_ideal[i] else f"Dataset Sample"
        ds_str = f"\nBest in Dataset: {best_dataset_mat} (Abs={best_dataset_abs:.3f})" if (best_dataset_abs is not None and is_ideal[i]) else ""
        ax_p.set_title(f"{title_prefix} (P-Pol) | Predicted: {pred_mat_name}\nTorcwa Abs={rcwa_avg_abs:.3f} | Surr Abs={surr_avg_abs:.3f}{ds_str}")

        # 3. S-Pol Spectra
        ax_s = axes[i, 1]
        ax_s.plot(WAVELENGTHS, curve[i, n_wl_half:].cpu().numpy(), "k-", lw=2.5, label="Target", zorder=10)
        if best_dataset_curve is not None and is_ideal[i]:
            ax_s.plot(WAVELENGTHS, best_dataset_curve[n_wl_half:].numpy(), color=c_dataset,
                      linestyle="-", lw=1.8, alpha=0.75, zorder=9)
        ax_s.plot(WAVELENGTHS, surrogate_preds[i, n_wl_half:].cpu().numpy(), linestyle="--", color=c_surr, lw=2.0, label="Surrogate")
        ax_s.plot(WAVELENGTHS, rcwa_s, linestyle="-", color=c_physics, lw=2.0, label="Torcwa")
        ax_s.set_xlim(300, 1100); ax_s.set_ylim(-0.05, 1.05)
        ax_s.set_xlabel("Wavelength (nm) — S-Pol")
        ax_s.set_ylabel("Absorptance")
        if bands:
            for bmin, bmax in bands:
                ax_s.axvspan(bmin, bmax, color="gray", alpha=0.2)
        ax_s.set_title(f"{title_prefix} (S-Pol) | Predicted: {pred_mat_name}\nTorcwa Abs={rcwa_avg_abs:.3f} | Surr Abs={surr_avg_abs:.3f}{ds_str}")
        
        # 4. Grating Profile
        ax_g = axes[i, 2]
        amps = px[:, 0].cpu().numpy()
        phases = px[:, 1].cpu().numpy()
        
        grating_height = 2.0 * amps.sum() + 1e-9
        arg = 2.0 * np.pi * harmonic_idx[:, None] * r_grid[None, :] / 1000.0 - phases[:, None]
        cosines = amps[:, None] * np.cos(arg)
        prof = grating_height / 2.0 + cosines.sum(axis=0)
        
        ax_g.fill_between(r_grid, 0, prof, color=c_profile, alpha=0.6)
        ax_g.plot(r_grid, prof, color=c_profile, lw=1.5)
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
        ax_h.bar(x_pos, amps, color=c_amp, edgecolor="black")
        ax_h.set_ylabel("Amplitude (nm)", color=c_amp)
        ax_h.tick_params(axis='y', labelcolor=c_amp)
        
        ax_p2 = ax_h.twinx()
        ax_p2.plot(x_pos, phases, 'o', color=c_phase, markersize=8)
        ax_p2.set_ylabel("Phase (rad)", color=c_phase)
        ax_p2.tick_params(axis='y', labelcolor=c_phase)
        ax_p2.set_ylim(-0.5, 2*np.pi + 0.5)
        
        ax_h.set_title("Harmonics Amplitudes & Phases")
        ax_h.set_xticks(x_pos)
        ax_h.set_xticklabels([f"n={n}" for n in x_pos])
        
        if i == n_samples - 1:
            ax_p.set_xlabel("Wavelength (nm)")
            ax_s.set_xlabel("Wavelength (nm)")
            ax_g.set_xlabel("x (nm)")
            ax_h.set_xlabel("Harmonic Index")
            
    if forward_model_name:
        fig.suptitle(f"Model: {model_name} (Frozen Surrogate: {Path(forward_model_name).stem})", fontsize=20, y=1.02)
    else:
        fig.suptitle(f"Model: {model_name}", fontsize=20, y=1.02)
        
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved Dashboard: {save_path}")
    return np.mean(mse_list)

def plot_inverse_loss_curves(all_history: dict, save_path: str):
    if not all_history:
        return
    
    # Filter out models that didn't return a valid history dict
    valid_history = {k: v for k, v in all_history.items() if v and "train_loss" in v}
    n_models = len(valid_history)
    
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False, layout="constrained")
    axes = axes[0]

    for ax, (name, hist) in zip(axes, valid_history.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], label="Train", alpha=0.8)
        
        if "val_loss" in hist:
            ax.semilogy(epochs, hist["val_loss"], label="Val", alpha=0.8)
            
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Inverse Training Loss Curves", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "evaluation" / "inverse"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    mat_names = stats["materials"]
    print(f"Loaded Stats: n_continuous={n_continuous}, n_wavelengths={n_wavelengths}, materials={mat_names}")

    batch = generate_test_batch(stats)
    


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
    all_history = {}
    
    # Load Models
    for model_file in ["tandem.pt", "generative_tandem.pt", "cvae.pt"]:
        path = ckpt_dir / model_file
        if path.exists():
            inv_name = path.stem
            
            # Extract the precise forward model name used during training (or override)
            if getattr(args, "force_forward_model", None):
                fwd_name = args.force_forward_model
            else:
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    fwd_name = ckpt.get("forward_model_name", "")
                    if not fwd_name and "metadata" in ckpt:
                        fwd_name = ckpt["metadata"].get("forward_model_name", "")
                except Exception:
                    fwd_name = ""
                
            forward_model = None
            if fwd_name:
                try:
                    forward_model, _, _ = load_forward_model(
                        ckpt_dir / fwd_name,
                        n_continuous=n_continuous,
                        n_wavelengths=n_wavelengths,
                        n_harmonics=n_harmonics
                    )
                    forward_model.eval()
                    forward_model.to(device)
                    print(f"\n=> Loaded Frozen Surrogate for {inv_name}: {fwd_name}")
                except Exception as e:
                    print(f"\n=> ERROR: Failed to load specified surrogate '{fwd_name}': {e}")
            else:
                print(f"\n=> WARNING: {inv_name} metadata missing 'forward_model_name'.")

            try:
                model, hist, metadata = load_inverse_model(
                    path, forward_model=forward_model, dataset_stats=stats, n_continuous=n_continuous, n_wavelengths=n_wavelengths
                )
                model.eval()
                model.to(device)
            except Exception as e:
                print(f"  Failed to load {inv_name}: {e}")
                continue

            inverse_models[inv_name] = model
            all_history[inv_name] = hist
            model._fwd_name_used = fwd_name
            print(f"Loaded: {inv_name}")

    # Retrieve RCWA Config
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_names[0]}" / "batch_0000.pt"
    if first_batch_file.exists():
        rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    else:
        rcwa_config_dict = {}

    # --- Parse bands ---
    bands: list[tuple[float, float]] = []
    if args.bands:
        if len(args.bands) % 2 != 0:
            print("Error: --bands must have an even number of arguments (min max pairs).")
            return
        for i in range(0, len(args.bands), 2):
            bands.append((args.bands[i], args.bands[i + 1]))

    bands_str = "_".join([f"{int(b[0])}-{int(b[1])}" for b in bands]) if bands else "full_spectrum"
    if bands:
        print(f"\nBands: {bands}")
        print(f"Building ideal band step-function target (1.0 in-band, 0.0 out-of-band)")

    # --- Dataset baseline (best curve in bands) ---
    best_dataset_curve: torch.Tensor | None = None
    best_dataset_abs: float | None = None
    best_dataset_mat: str | None = None
    if bands:
        print("\nScanning dataset for best raw performance in bands...")
        try:
            baseline_res, _ = get_dataset_baseline(ckpt_dir, bands)
            best_score = -1.0
            for mat, res in baseline_res.items():
                score = res["avg_abs"].max().item()
                if score > best_score:
                    best_score = score
                    best_dataset_abs = score
                    best_dataset_mat = mat
                    best_dataset_curve = res["targets"][res["avg_abs"].argmax()].clone()
            print(f"  Best dataset structure: {best_dataset_mat}, Avg Abs={best_dataset_abs:.4f}")
        except Exception as e:
            import traceback
            print(f"  Warning: could not load dataset baseline:")
            traceback.print_exc()

    print("\nGenerating comprehensive model dashboards...")

    if all_history:
        plot_inverse_loss_curves(all_history, str(eval_dir / "inverse_loss_curves.png"))

    # --- Build targets ---
    # Always include 1 real sample from the test batch
    real_targets = batch["target"][:1]

    if bands:
        # Step function: 1.0 inside any band, 0.0 outside — for both p and s polarisations
        step = np.zeros(len(WAVELENGTHS))
        for bmin, bmax in bands:
            step[(WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)] = 1.0
        step_both_pol = np.concatenate([step, step])          # (n_wavelengths,)
        band_target_1 = torch.from_numpy(step_both_pol).float().unsqueeze(0)
    else:
        band_target_1 = torch.ones(1, n_wavelengths)          # broadband fallback

    all_metrics = {}

    for name, m in inverse_models.items():
        # Deterministic models (plain Tandem): only 1 band-target row
        # Stochastic models (GenTandem, CVAE): 2 band-target rows (different samples)
        is_deterministic = (
            hasattr(m, "inverse_decoder")
            and not hasattr(m, "sample_diverse_designs")
            and not hasattr(m, "spectrum_encoder")
        )
        if is_deterministic:
            targets_to_eval = torch.cat([real_targets, band_target_1], dim=0)
            is_ideal = [False, False, True]
        else:
            targets_to_eval = torch.cat([real_targets, band_target_1, band_target_1], dim=0)
            is_ideal = [False, False, True, True]

        save_path = str(eval_dir / f"dashboard_{name}_{bands_str}.png")
        avg_mse = plot_model_dashboard(
            model_name=name,
            inv_model=m,
            forward_model=forward_model,
            targets=targets_to_eval,
            is_ideal=is_ideal,
            rcwa_config_dict=rcwa_config_dict,
            stats=stats,
            save_path=save_path,
            n_wavelengths=n_wavelengths,
            bands=bands if bands else None,
            best_dataset_curve=best_dataset_curve,
            best_dataset_abs=best_dataset_abs,
            best_dataset_mat=best_dataset_mat,
            requested_material=args.material,
            forward_model_name=getattr(m, "_fwd_name_used", ""),
        )
        all_metrics[name] = {"mse": float(avg_mse)}

    metrics_path = eval_dir / f"inverse_metrics_{bands_str}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll evaluation outputs and metrics saved to: {eval_dir}")

if __name__ == "__main__":
    main()
