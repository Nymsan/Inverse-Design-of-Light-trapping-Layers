#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import N_MATERIALS, build_profile, MATERIAL_LIBRARY
from Scripts.evaluate_dataset_baseline import get_dataset_baseline
from Utils.surrogate_optimization import BatchedSurrogateOptimizer, recover_geometry_from_profile
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.checkpoint import load_forward_model

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Batched Surrogate Optimizer")
    p.add_argument("--ckpt_dir", default="Checkpoints/Si_TiO2_Si3N4", help="Path to checkpoint dir")
    p.add_argument("--mode", choices=["geometry", "profile"], default="geometry", help="Optimization mode")
    p.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to optimize, e.g., --bands 500 750 800 900")
    p.add_argument("--restarts", type=int, default=5000, help="Number of random restarts per material")
    p.add_argument("--steps", type=int, default=300, help="Optimization steps")
    p.add_argument("--max_inc_deg", type=float, default=30.0, help="Maximum incident angle in degrees (default: 30)")
    p.add_argument("--recovered_harmonics", type=int, default=10, help="Number of harmonics to recover via FFT in profile mode (default: 10)")
    return p.parse_args()

def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        print(f"Stats not found at {stats_path}")
        return
        
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    
    geo_min = stats["geo_min"]
    geo_max = stats["geo_max"]
    
    trained_mat_names = list(stats["materials"].keys())
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{trained_mat_names[0]}" / "batch_0000.pt"
    if first_batch_file.exists():
        rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    else:
        rcwa_config_dict = {}
    
    trained_mat_names = list(stats["materials"].keys())
    valid_mat_indices = [MATERIAL_LIBRARY[name] for name in trained_mat_names]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn_path = ckpt_dir / "spatial_cnn.pt"
    if not cnn_path.exists():
        print(f"{cnn_path} not found! Please train forward models first.")
        return
        
    model, _, _ = load_forward_model(
        cnn_path, n_continuous=stats["n_continuous"], n_wavelengths=stats["n_wavelengths"], n_harmonics=stats["n_harmonics"]
    )
    model = model.to(device)
    model.eval()
    
    # Parse bands
    bands = []
    if args.bands:
        if len(args.bands) % 2 != 0:
            print("Error: --bands must have an even number of arguments (min max pairs)")
            return
        for i in range(0, len(args.bands), 2):
            bands.append((args.bands[i], args.bands[i+1]))
            
    print(f"Running Batched Surrogate Optimization (Mode: {args.mode})")
    print(f"Bands: {bands if bands else 'Full Spectrum 300-1100nm'}")
    
    print(f"Clamping Incident Angle to {args.max_inc_deg} degrees")
    
    opt = BatchedSurrogateOptimizer(model, geo_min, geo_max, device, nx=128, max_inc_deg=args.max_inc_deg)
    
    if args.mode == "geometry":
        res = opt.optimize_geometry(bands, n_restarts=args.restarts, steps=args.steps, lr=0.05, allowed_materials=valid_mat_indices)
        geo = res["best_geometry"]
        profile, h_tensor, inc_tensor = build_profile(geo.unsqueeze(0), stats["n_harmonics"], nx=128)
        profile = profile[0]
        h_val = geo[-2].item()
        inc_ang = geo[-1].item()
        
    elif args.mode == "profile":
        res = opt.optimize_profile(bands, n_restarts=args.restarts, steps=args.steps, lr=5.0, allowed_materials=valid_mat_indices)
        profile = res["best_profile"]
        geo = recover_geometry_from_profile(profile, res["best_h"], res["best_inc_ang"], nx=128)
        h_val = res["best_h"].item()
        inc_ang = res["best_inc_ang"].item()
        
    mat_idx = res["best_material"]
    mat_name = list(MATERIAL_LIBRARY.keys())[mat_idx]
    surrogate_curve = res["best_curve"]
    target_curve = res["target"]
    mask = res["mask"]
    
    print(f"Optimization finished! Best Surrogate Absorptance: {res['best_loss']:.4f}")
    print(f"Selected Material: {mat_name}")
    print(f"Height: {h_val:.2f} nm, Incident Angle: {inc_ang:.2f} deg")
    
    # Plotting
    out_dir = ckpt_dir / "evaluation" / "surrogate_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    bands_str = "_".join([f"{int(b[0])}-{int(b[1])}" for b in bands]) if bands else "full_spectrum"
    
    # Plot history
    history = res["history"].numpy()
    plt.figure(figsize=(8, 6))
    for i, mat_idx in enumerate(res["allowed_materials"]):
        mat_name_h = list(MATERIAL_LIBRARY.keys())[mat_idx]
        plt.plot(history[:, i], label=mat_name_h)
    plt.xlabel('Steps')
    plt.ylabel('Max Surrogate Absorptance')
    plt.legend()
    plt.title('Optimization History (Max Absorptance per Material)')
    plt.tight_layout()
    plt.savefig(out_dir / f"optimization_history_{args.mode}_{bands_str}.png")
    plt.close()
    
    n_results = len(res["top_results"])
    fig, axes = plt.subplots(n_results, 4, figsize=(24, 6 * n_results), layout="constrained")
    if n_results == 1:
        axes = np.expand_dims(axes, axis=0)
        
    metrics_list = []
    
    print("\nScanning dataset for best raw performance in bands...")
    try:
        baseline_res, _ = get_dataset_baseline(ckpt_dir, bands)
        best_dataset_abs = {m: r["avg_abs"].max().item() for m, r in baseline_res.items()}
        best_dataset_target = {m: r["targets"][r["avg_abs"].argmax()].clone() for m, r in baseline_res.items()}
        
        for m, score in best_dataset_abs.items():
            if score > 0:
                print(f"Best Dataset Structure (Mat: {m}) Average Absorptance: {score:.4f}")
    except Exception as e:
        print(f"Warning: Could not load dataset baseline ({e})")
        best_dataset_abs = {}
        best_dataset_target = {}
    
    print(f"\nRunning Torcwa simulations for {n_results} top structures...")
    for idx, r in enumerate(tqdm(res["top_results"], desc="Verifying with Torcwa")):
        mat_name = list(MATERIAL_LIBRARY.keys())[r["material_idx"]]
        if args.mode == "geometry":
            geo = r["geometry"]
            prof_tensor, h_tensor, inc_tensor = build_profile(geo.unsqueeze(0), stats["n_harmonics"], nx=128)
            profile_np = prof_tensor[0].numpy()
            h_val = geo[-2].item()
            inc_ang = geo[-1].item()
        else:
            profile_np = r["profile"].numpy()
            h_val = r["h"].item()
            inc_ang = r["inc_ang"].item()
            geo = recover_geometry_from_profile(r["profile"], r["h"], r["inc_ang"], nx=128, n_harmonics=args.recovered_harmonics)
        
        n_fourier = len(geo) - 2
        px = geo[:n_fourier].unsqueeze(0).cpu()
        
        base_config = RCWAConfig(**rcwa_config_dict)
        base_config.h = h_val
        base_config.inc_ang = (inc_ang + 1e-3) * np.pi / 180.0
        base_config.azi_ang = 1e-3 * np.pi / 180.0
        if mat_name.endswith("_Ag"):
            base_config.grating_material = mat_name[:-3]
            base_config.reflector_type = 'Ag'
        else:
            base_config.grating_material = mat_name
            base_config.reflector_type = 'pec'
            
        WAVELENGTHS = np.linspace(300, 1100, stats["n_wavelengths"] // 2)
        A_film, _ = get_absorptance_curve(
            params_x=px,
            params_y=None,
            wavelengths=torch.from_numpy(WAVELENGTHS).double(),
            config=base_config,
            show_progress=False
        )
        rcwa_p = A_film[:, 0].cpu().numpy()
        rcwa_s = A_film[:, 1].cpu().numpy()
        
        sim_np = np.concatenate([rcwa_p, rcwa_s])
        target_np = target_curve.cpu().numpy()
        rcwa_mae = float(np.mean(np.abs(target_np - sim_np)))
        
        if bands:
            mask = np.zeros(len(WAVELENGTHS), dtype=bool)
            for bmin, bmax in bands:
                mask |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
        else:
            mask = np.ones(len(WAVELENGTHS), dtype=bool)

        avg_abs_p = np.mean(rcwa_p[mask])
        avg_abs_s = np.mean(rcwa_s[mask])
        rcwa_avg_abs = float((avg_abs_p + avg_abs_s) / 2.0)
        
        metrics = {
            "rank": (idx % 2) + 1,
            "surrogate_loss": r["loss"],
            "rcwa_mae": rcwa_mae,
            "rcwa_avg_abs": rcwa_avg_abs,
            "material": mat_name,

            "h": h_val,
            "inc_ang": inc_ang,
            "geometry": geo.tolist()
        }
        metrics_list.append(metrics)
        
        cmap = plt.cm.viridis
        
        print(f"  -> Rank {(idx%2)+1} [{mat_name}]: Torcwa Avg Abs = {rcwa_avg_abs:.4f} (Surrogate Predicted Abs = {r['loss']:.4f})")
        
        c_surr, c_physics = cmap(0.5), cmap(0.8)
        
        ax_row = axes[idx]
        
        best_target_for_mat = best_dataset_target.get(mat_name)
        best_abs_for_mat = best_dataset_abs.get(mat_name, -1.0)
        
        # P-pol
        ax = ax_row[0]
        ax.plot(WAVELENGTHS, target_np[:len(WAVELENGTHS)], "k-", lw=2, label="Target")
        if best_target_for_mat is not None:
            ax.plot(WAVELENGTHS, best_target_for_mat[:len(WAVELENGTHS)].numpy(), color="k", linestyle=":", lw=2, label="Best Dataset", alpha=0.6)
        ax.plot(WAVELENGTHS, r["curve"][:len(WAVELENGTHS)].numpy(), linestyle="--", color=c_surr, lw=2, label="Surrogate")
        ax.plot(WAVELENGTHS, rcwa_p, linestyle="-", color=c_physics, lw=2, label="Torcwa Physics")
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.2)
        dataset_str = f" | Dataset Abs={best_abs_for_mat:.3f}" if best_target_for_mat is not None else ""
        ax.set_title(f"Rank {(idx%2)+1}: {mat_name} (P-Pol)\nTorcwa Abs={rcwa_avg_abs:.3f} | Surr Abs={r['loss']:.4f}{dataset_str}", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0: ax.legend(fontsize=9)
        
        # S-pol
        ax = ax_row[1]
        ax.plot(WAVELENGTHS, target_np[len(WAVELENGTHS):], "k-", lw=2, label="Target")
        if best_target_for_mat is not None:
            ax.plot(WAVELENGTHS, best_target_for_mat[len(WAVELENGTHS):].numpy(), color="k", linestyle=":", lw=2, label="Best Dataset", alpha=0.6)
        ax.plot(WAVELENGTHS, r["curve"][len(WAVELENGTHS):].numpy(), linestyle="--", color=c_surr, lw=2, label="Surrogate")
        ax.plot(WAVELENGTHS, rcwa_s, linestyle="-", color=c_physics, lw=2, label="Torcwa Physics")
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.2)
        dataset_str2 = f" | Dataset Abs={best_abs_for_mat:.3f}" if best_target_for_mat is not None else ""
        ax.set_title(f"Rank {(idx%2)+1}: {mat_name} (S-Pol)\nTorcwa Abs={rcwa_avg_abs:.3f} | Surr Abs={r['loss']:.4f}{dataset_str2}", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        
        # Structure cross-section
        ax = ax_row[2]
        xs = np.linspace(0, rcwa_config_dict.get("grating_period", 1000), 128)
        ax.plot(xs, profile_np, "k-", lw=1.5)
        ax.fill_between(xs, 0, profile_np, color=cmap(0.7), alpha=0.5)
        if args.mode == "profile":
            n_harm_recovered = (len(geo) - 2) // 2
            rec_prof, _, _ = build_profile(geo.unsqueeze(0).cpu(), n_harm_recovered, nx=128)
            ax.plot(xs, rec_prof[0].numpy(), "c--", lw=1.5, label="FFT Recovered")
            if idx == 0: ax.legend(fontsize=8)
        ax.set_title(f"Structure Cross-Section\nHeight={h_val:.0f}nm, Inc={inc_ang:.1f}°", fontsize=13)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Height (nm)")

        # Harmonics amplitudes & phases
        ax_h = ax_row[3]
        n_harm = (len(geo) - 2) // 2
        geo_cpu = geo.cpu()
        amps_geo   = geo_cpu[:n_harm].numpy()           # raw amplitudes (nm)
        phases_geo = geo_cpu[n_harm:2*n_harm].numpy()   # raw phases (rad)
        x_pos = np.arange(1, n_harm + 1)
        c_amp   = cmap(0.3)
        c_phase = cmap(0.9)
        ax_h.bar(x_pos, amps_geo, color=c_amp, edgecolor="black")
        ax_h.set_ylabel("Amplitude (nm)", color=c_amp)
        ax_h.tick_params(axis='y', labelcolor=c_amp)
        ax_h.set_xlabel("Harmonic index")
        ax_p2 = ax_h.twinx()
        ax_p2.plot(x_pos, phases_geo, 'o', color=c_phase, markersize=8)
        ax_p2.set_ylabel("Phase (rad)", color=c_phase)
        ax_p2.tick_params(axis='y', labelcolor=c_phase)
        ax_p2.set_ylim(-0.5, 2 * np.pi + 0.5)
        ax_h.set_title(f"Harmonics Amplitudes & Phases", fontsize=13)
        
    out_path = out_dir / f"bgd_optimization_{args.mode}_{bands_str}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    metrics_path = out_dir / f"surrogate_optimization_metrics_{args.mode}_{bands_str}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
