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
from Utils.utils import sun_weights, get_jsc_scaling_factor

from Utils.models import N_MATERIALS, build_profile, MATERIAL_LIBRARY, SIREN
from Scripts.evaluate_dataset_baseline import get_dataset_baseline
from Utils.surrogate_optimization import BatchedSurrogateOptimizer
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.checkpoint import load_forward_model, get_best_forward_model

# Styling
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

MATERIAL_COLORS = {
    "Si": "tab:blue",
    "TiO2": "tab:orange",
    "Si3N4": "tab:green"
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Batched Surrogate Optimizer")
    p.add_argument("--ckpt_dir", default="Checkpoints/Si_TiO2_Si3N4", help="Path to checkpoint dir")
    p.add_argument("--mode", choices=["geometry", "de"], default="geometry", help="Optimization mode")
    p.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to optimize, e.g., --bands 500 750 800 900")
    p.add_argument("--h_val", nargs="+", type=float, help="Target height in nm, or range (min max) (fixes/bounds height during evaluation)")
    p.add_argument("--inc_val", type=float, default=None, help="Target incident angle in degrees (fixes angle during evaluation)")
    p.add_argument("--restarts", type=int, default=500, help="Number of gradient descent runs (top-k from dense sampling) per material")
    p.add_argument("--dense_samples", type=int, default=10000000, help="Number of dense samples to evaluate before gradient descent")
    p.add_argument("--steps", type=int, default=500, help="Optimization steps")
    p.add_argument("--save_dir", type=str, default="Checkpoints/Optimization_Eval")
    p.add_argument("--al_iter", type=int, default=-1, help="Active learning iteration of surrogate to use (-1 for latest, 0 for base)")
    p.add_argument("--max_inc_deg", type=float, default=None, help="Maximum incident angle in degrees (default: None, takes max from dataset)")
    p.add_argument("--n_harmonics", type=int, default=None, help="Extended number of harmonics to optimize. If not set, defaults to dataset n_harmonics.")
    p.add_argument("--order_N", type=int, default=None, help="RCWA Order N override for Torcwa evaluation")
    p.add_argument("--height_per_layer", type=float, default=None, help="RCWA height per layer override for Torcwa evaluation")

    p.add_argument("--top_k", type=int, default=1, help="Number of top structures to show and simulate per material")
    p.add_argument("--force_forward_model", type=str, default=None, help="Force load a specific forward model (e.g. 'siren.pt')")
    p.add_argument("--expand_amps", type=float, default=None, help="Temporarily expand the maximum amplitude bounds (e.g., to 25.0 nm)")
    p.add_argument("--optimize_jsc", action="store_true", help="Optimize Short-Circuit Current (Jsc) weighted by AM1.5G photon flux and cos(inc_ang), rather than average absorptance.")
    p.add_argument("--eval_resolution", type=int, default=None, help="Number of wavelengths between 300-1100 nm used for the final Torcwa validation curve. Default: dataset n_wavelengths // 2.")
    p.add_argument("--siren_search_resolution", type=int, default=None, help="(SIREN only) Number of wavelengths queried per forward pass during surrogate optimisation. Default: model's trained n_wavelengths // 2.")
    return p.parse_args()

def get_folder_name(args) -> str:
    parts = []
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            parts.append(f"h{int(args.h_val[0])}-{int(args.h_val[1])}")
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            parts.append(f"h{int(h_target)}")
    if args.inc_val is not None:
        parts.append(f"inc{args.inc_val}")
    if args.bands:
        bands_str = "_".join([f"{int(args.bands[i])}-{int(args.bands[i+1])}" for i in range(0, len(args.bands), 2)])
        parts.append(f"bands{bands_str}")
        
    if getattr(args, "optimize_jsc", False):
        parts.append("jsc")
        
    if getattr(args, "eval_resolution", None) is not None:
        parts.append(f"res{args.eval_resolution}")
        
    return "_".join(parts) if parts else "all_data"

def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        print(f"Stats not found at {stats_path}")
        return
        
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo_min = stats["geo_min"].to(device)
    geo_max = stats["geo_max"].to(device)
    
    if args.expand_amps is not None:
        n_harmonics = stats["n_harmonics"]
        # In models.py, geometry is packed as [amp0, phase0, amp1, phase1...]
        # Amplitudes are at even indices: 0, 2, 4... up to 2*n_harmonics
        geo_max[0:2*n_harmonics:2] = args.expand_amps
        print(f"[*] Expanded Amplitude Bounds to {args.expand_amps} nm for Surrogate Optimization!")
        
    n_harmonics_opt = stats["n_harmonics"]
    if args.n_harmonics is not None and args.n_harmonics > stats["n_harmonics"]:
        n_harmonics_opt = args.n_harmonics
        n_extra = n_harmonics_opt - stats["n_harmonics"]
        print(f"[*] Extending optimized geometry parameters from {stats['n_harmonics']} to {n_harmonics_opt} harmonics")
        
        extra_min = torch.zeros(n_extra * 2, device=device)
        extra_max = torch.zeros(n_extra * 2, device=device)
        
        # Extended phases bounded 0 to 2pi
        extra_max[1::2] = 2 * torch.pi
        
        # Extended amps bounded to expand_amps (or 0 if not provided, meaning they won't do much)
        if args.expand_amps is not None:
            extra_max[0::2] = args.expand_amps
        else:
            # Match the highest amplitude of the original bounds if no expand_amps is given
            extra_max[0::2] = geo_max[2*stats["n_harmonics"] - 2]
            
        geo_min = torch.cat([geo_min[:-2], extra_min, geo_min[-2:]])
        geo_max = torch.cat([geo_max[:-2], extra_max, geo_max[-2:]])
        
    # --- Resize bounds if h_val or inc_val exceed dataset bounds ---
    if args.h_val is not None:
        if isinstance(args.h_val, list) and len(args.h_val) == 2:
            h_target_min, h_target_max = args.h_val
            if h_target_max > geo_max[-2]: geo_max[-2] = h_target_max
            if h_target_min < geo_min[-2]: geo_min[-2] = h_target_min
        else:
            h_target = args.h_val[0] if isinstance(args.h_val, list) else args.h_val
            if h_target > geo_max[-2]: geo_max[-2] = h_target
            if h_target < geo_min[-2]: geo_min[-2] = h_target
            
    if args.inc_val is not None:
        if args.inc_val > geo_max[-1]: geo_max[-1] = args.inc_val
        if args.inc_val < geo_min[-1]: geo_min[-1] = args.inc_val
    # ---------------------------------------------------------------
    
    trained_mat_names = stats["materials"]
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{trained_mat_names[0]}" / "batch_0000.pt"
    if first_batch_file.exists():
        rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    else:
        rcwa_config_dict = {}
    
    trained_mat_names = stats["materials"]
    valid_mat_indices = [MATERIAL_LIBRARY[name] for name in trained_mat_names]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the best forward model
    model, fwd_name, test_loss = get_best_forward_model(
        ckpt_dir, n_continuous=stats["n_continuous"], n_wavelengths=stats["n_wavelengths"], n_harmonics=stats["n_harmonics"], al_iter=args.al_iter, force_model_name=args.force_forward_model
    )
    if model is None:
        print("\n=> ERROR: No forward models found in Checkpoints directory!")
        return
    
    print(f"Loaded {Path(fwd_name).name} for surrogate optimization (test loss: {test_loss:.4f})")
    
    # --- Resolution overrides ---
    is_siren = isinstance(model, SIREN)
    
    # Torcwa eval resolution: number of wavelengths in the final physics simulation
    eval_n_wl = args.eval_resolution if args.eval_resolution is not None else stats["n_wavelengths"] // 2
    
    # Surrogate search resolution: wavelengths queried per SIREN forward pass during optimisation
    # For non-SIREN models the output size is fixed, so this flag is ignored.
    search_n_wl = None  # None means "use model default" (passed as n_wavelengths to optimizer)
    if args.siren_search_resolution is not None:
        if is_siren:
            search_n_wl = args.siren_search_resolution * 2  # _get_target_and_mask expects total (p+s)
            model.seq_len = args.siren_search_resolution
            print(f"[*] SIREN search resolution overridden to {args.siren_search_resolution} wavelengths per polarisation")
        else:
            print(f"[!] --siren_search_resolution ignored: loaded model is not a SIREN (it is {type(model).__name__})")
    
    print(f"\nRunning Surrogate Optimization ({args.mode} mode) ...")
    
    # Extract al_iterations from history if available
    al_iterations = model.history.get("al_iterations", 0) if hasattr(model, "history") else 0
    title_suffix = f" (Surrogate: {Path(fwd_name).stem})"
    
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
    
    if args.inc_val is not None:
        print(f"[*] Pinned Incident Angle to exactly {args.inc_val:.2f} degrees")
    else:
        actual_max_inc = args.max_inc_deg if args.max_inc_deg is not None else geo_max[-1].item()
        print(f"Bounded Incident Angle search up to {actual_max_inc:.2f} degrees")
    
    opt = BatchedSurrogateOptimizer(
        model, geo_min, geo_max, n_harmonics=n_harmonics_opt,
        nx=128, device=device, max_inc_deg=args.max_inc_deg, h_val=args.h_val, inc_val=args.inc_val
    )
    if args.mode == "geometry":
        res = opt.optimize_geometry(
            bands=bands,
            n_restarts=args.restarts,
            n_dense_samples=args.dense_samples,
            steps=args.steps,
            lr=0.075,
            allowed_materials=valid_mat_indices,
            top_k=args.top_k,
            show_progress=True,
            optimize_jsc=args.optimize_jsc,
            override_n_wavelengths=search_n_wl
        )
        geo = res["best_geometry"]
        profile, h_tensor, inc_tensor = build_profile(geo.unsqueeze(0), n_harmonics_opt, nx=128)
        profile = profile[0]
        h_val = geo[-2].item()
        inc_ang = geo[-1].item()
        
    elif args.mode == "de":
        res = opt.optimize_de(bands, pop_size=args.restarts, generations=args.steps, F=0.8, CR=0.9, allowed_materials=valid_mat_indices, top_k=args.top_k, optimize_jsc=args.optimize_jsc, override_n_wavelengths=search_n_wl)
        geo = res["best_geometry"]
        profile, h_tensor, inc_tensor = build_profile(geo.unsqueeze(0), n_harmonics_opt, nx=128)
        profile = profile[0]
        h_val = geo[-2].item()
        inc_ang = geo[-1].item()
        
    mat_idx = res["best_material"]
    mat_name = list(MATERIAL_LIBRARY.keys())[mat_idx]
    surrogate_curve = res["best_curve"]
    target_curve = res["target"]
    mask = res["mask"]
    
    print(f"Optimization finished! Best Surrogate Absorptance: {res['best_loss']:.4f}")
    print(f"Selected Material: {mat_name}")
    print(f"Height: {h_val:.2f} nm, Incident Angle: {inc_ang:.2f} deg")
    
    # Plotting
    folder_name = get_folder_name(args)
    if folder_name == "all_data":
        folder_name = f"all_data_{Path(fwd_name).stem}"
    else:
        folder_name = f"{folder_name}_{Path(fwd_name).stem}"
    out_dir = ckpt_dir / "evaluation" / "surrogate_optimization" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bands_str = "_".join([f"{int(b[0])}-{int(b[1])}" for b in bands]) if bands else "full_spectrum"
    
    # Scan dataset for best raw performance in bands
    print("\nScanning dataset for best raw performance in bands...")
    try:
        baseline_res, _ = get_dataset_baseline(ckpt_dir, bands=bands, h_val=args.h_val, inc_val=args.inc_val, optimize_jsc=args.optimize_jsc)
        best_dataset_abs = {m: r["metric"].max().item() for m, r in baseline_res.items()}
        best_dataset_target = {m: r["targets"][r["metric"].argmax()].clone() for m, r in baseline_res.items()}
        best_dataset_geo = {m: r["geos"][r["metric"].argmax()].clone() for m, r in baseline_res.items()}
        
        for m, score in best_dataset_abs.items():
            print(f"Dataset Best {m}: {score:.4f}")
    except Exception as e:
        print(f"Could not load dataset baseline: {e}")
        baseline_res = None
        best_dataset_abs = {}
        best_dataset_target = {}
        best_dataset_geo = {}
        
    # Plot history
    history = res["history"].numpy()
    plt.figure(figsize=(8, 6))
    for i, mat_idx in enumerate(res["allowed_materials"]):
        mat_name_h = list(MATERIAL_LIBRARY.keys())[mat_idx]
        color = plt.cm.tab10(i % 10)
        plt.plot(history[:, i], label=mat_name_h, color=color)
        if mat_name_h in best_dataset_abs:
            plt.axhline(best_dataset_abs[mat_name_h], color=color, linestyle="--", alpha=0.5, label=f"Dataset Max ({mat_name_h})")
            
    plt.xlabel('Steps')
    plt.ylabel('Max Surrogate Absorptance')
    plt.legend()
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig(out_dir / f"optimization_history_{args.mode}.png")
    plt.close()
    
    n_results = len(res["top_results"])
    fig, axes = plt.subplots(n_results, 5, figsize=(38, 7 * n_results), squeeze=False)
        
    metrics_list = []
    
    print(f"\nRunning Torcwa simulations for {n_results} top structures...")
    for idx, r in enumerate(tqdm(res["top_results"], desc="Verifying with Torcwa")):
        mat_name = list(MATERIAL_LIBRARY.keys())[r["material_idx"]]
        geo = r["geometry"]
        prof_tensor, h_tensor, inc_tensor = build_profile(geo.unsqueeze(0), n_harmonics_opt, nx=128)
        profile_np = prof_tensor[0].numpy()
        h_val = geo[-2].item()
        inc_ang = geo[-1].item()
        
        n_fourier = len(geo) - 2
        px = geo[:n_fourier].view(-1, 2).cpu()
        
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
            
        if args.order_N is not None:
            base_config.order_N = args.order_N
        if args.height_per_layer is not None:
            base_config.height_per_layer = args.height_per_layer
            
        WAVELENGTHS = np.linspace(300, 1100, eval_n_wl)
        A_film, _ = get_absorptance_curve(
            params_x=px,
            params_y=None,
            wavelengths=torch.from_numpy(WAVELENGTHS).double(),
            config=base_config,
            show_progress=True
        )
        rcwa_p = A_film[:, 0].cpu().numpy()
        rcwa_s = A_film[:, 1].cpu().numpy()
        
        sim_np = np.concatenate([rcwa_p, rcwa_s])
        target_np = target_curve.cpu().numpy()
        
        if len(target_np) != len(sim_np):
            # Interpolate the surrogate curve to match Torcwa's evaluation grid
            n_target_wl = len(target_np) // 2
            target_wls = np.linspace(300, 1100, n_target_wl)
            target_p = target_np[:n_target_wl]
            target_s = target_np[n_target_wl:]
            
            interp_target_p = np.interp(WAVELENGTHS, target_wls, target_p)
            interp_target_s = np.interp(WAVELENGTHS, target_wls, target_s)
            target_np = np.concatenate([interp_target_p, interp_target_s])
            
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

        if getattr(args, "optimize_jsc", False):
            
            # Replicate the surrogate's Jsc math for the Torcwa arrays
            wls_t = torch.from_numpy(WAVELENGTHS).float()
            flux = sun_weights(wls_t) * wls_t
            mask_t = torch.from_numpy(mask).float()
            
            rcwa_avg_t = torch.from_numpy((rcwa_p + rcwa_s) / 2.0).float()
            jsc_val = (rcwa_avg_t * mask_t * flux).sum() * get_jsc_scaling_factor(len(WAVELENGTHS))
            
            # Cosine correction for incident angle
            cos_theta = np.cos(inc_ang * np.pi / 180.0)
            rcwa_val = float(jsc_val * cos_theta)
            
            metric_name = "Jsc"
            unit = " mA/cm²"
        else:
            rcwa_val = rcwa_avg_abs
            metric_name = "Abs"
            unit = ""
        
        rank = (idx % args.top_k) + 1
        metrics = {
            "rank": rank,
            "surrogate_model": fwd_name,
            "surrogate_loss": r["loss"],
            "rcwa_mae": rcwa_mae,   
            "rcwa_avg_abs": rcwa_avg_abs,
            "rcwa_jsc": rcwa_val if getattr(args, "optimize_jsc", False) else None,
            "material": mat_name,
            "h": h_val,
            "inc_ang": inc_ang,
            "geometry": geo.tolist()
        }
        metrics_list.append(metrics)
        
        cmap = plt.cm.viridis
        
        rank = (idx % args.top_k) + 1
        print(f"  -> Rank {rank} [{mat_name}]: Torcwa {metric_name} = {rcwa_val:.4f}{unit} (Surr {metric_name} = {r['loss']:.4f})")
        
        ax_row = axes[idx]
        
        best_target_for_mat = best_dataset_target.get(mat_name)
        best_geo_for_mat = best_dataset_geo.get(mat_name)
        
        # Original dataset targets for the best dataset point
        if best_target_for_mat is not None:
            n_dataset_wl = len(best_target_for_mat) // 2
            ds_wls = np.linspace(300, 1100, n_dataset_wl)
            ds_p_orig = best_target_for_mat[:n_dataset_wl].numpy()
            ds_s_orig = best_target_for_mat[n_dataset_wl:].numpy()
        else:
            ds_wls = ds_p_orig = ds_s_orig = None

        if best_geo_for_mat is not None:
            geo_data = best_geo_for_mat
            h_data = geo_data[-2].item()
            inc_data = geo_data[-1].item()
            
            if len(WAVELENGTHS) > n_dataset_wl:
                # Re-simulate the best dataset geometry at higher resolution
                ds_config = RCWAConfig(**rcwa_config_dict)
                ds_config.h = float(h_data)
                ds_config.inc_ang = (float(inc_data) + 1e-3) * np.pi / 180.0
                ds_config.azi_ang = 1e-3 * np.pi / 180.0
                if mat_name.endswith("_Ag"):
                    ds_config.grating_material = mat_name[:-3]
                    ds_config.reflector_type = 'Ag'
                else:
                    ds_config.grating_material = mat_name
                    ds_config.reflector_type = 'pec'
                    
                if args.order_N is not None:
                    ds_config.order_N = args.order_N
                if args.height_per_layer is not None:
                    ds_config.height_per_layer = args.height_per_layer
    
                n_harm_data = (len(geo_data) - 2) // 2
                px_data = geo_data[:2*n_harm_data].view(-1, 2).cpu()
                
                A_film_ds, _ = get_absorptance_curve(
                    params_x=px_data,
                    params_y=None,
                    wavelengths=torch.from_numpy(WAVELENGTHS).double(),
                    config=ds_config,
                    show_progress=False
                )
                bdt_p = A_film_ds[:, 0].cpu().numpy()
                bdt_s = A_film_ds[:, 1].cpu().numpy()
            else:
                bdt_p = np.interp(WAVELENGTHS, ds_wls, ds_p_orig)
                bdt_s = np.interp(WAVELENGTHS, ds_wls, ds_s_orig)
            
            # Recalculate dataset objective using the new higher resolution curve
            mask_ds = mask
            avg_abs_p_ds = np.mean(bdt_p[mask_ds])
            avg_abs_s_ds = np.mean(bdt_s[mask_ds])
            rcwa_avg_abs_ds = float((avg_abs_p_ds + avg_abs_s_ds) / 2.0)
            
            if getattr(args, "optimize_jsc", False):
                rcwa_avg_t_ds = torch.from_numpy((bdt_p + bdt_s) / 2.0).float()
                jsc_val_ds = (rcwa_avg_t_ds * mask_t * flux).sum() * get_jsc_scaling_factor(len(WAVELENGTHS))
                cos_theta_ds = np.cos(inc_data * np.pi / 180.0)
                best_abs_for_mat = float(jsc_val_ds * cos_theta_ds)
            else:
                best_abs_for_mat = rcwa_avg_abs_ds
                
        else:
            bdt_p = bdt_s = None
        
        c_surr, c_physics = cmap(0.5), cmap(0.8)
        c_dataset = cmap(0.2)
        c_amp = cmap(0.3)
        c_phase = cmap(0.9)
        
        surr_curve = r["curve"].cpu().numpy()
        n_surr_wl = len(surr_curve) // 2
        surr_wls = np.linspace(300, 1100, n_surr_wl)
        surr_p = surr_curve[:n_surr_wl]
        surr_s = surr_curve[n_surr_wl:]
        
        # P-pol
        ax = ax_row[0]
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.15)
        ax.plot(WAVELENGTHS, target_np[:len(WAVELENGTHS)], "k--", lw=3, label="Target")
        if bdt_p is not None:
            ax.plot(WAVELENGTHS, bdt_p, color=c_dataset, linestyle=":", lw=2, label="Best Dataset")
        ax.plot(surr_wls, surr_p, linestyle="-", color=c_surr, lw=3, label="Surrogate")
        ax.plot(WAVELENGTHS, rcwa_p, linestyle="-", color=c_physics, lw=2.5, label="Torcwa Physics")
        dataset_str = f" | Dataset {metric_name}={best_abs_for_mat:.3f}" if bdt_p is not None else ""
        ax.set_title(f"{mat_name} (P-Pol)\nTorcwa {metric_name}={rcwa_val:.2f}{unit} | Surr {metric_name}={r['loss']:.4f}{dataset_str}")
        ax.set_ylim(-0.05, 1.05)
        if idx == 0: ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # S-pol
        ax = ax_row[1]
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.15)
        ax.plot(WAVELENGTHS, target_np[len(WAVELENGTHS):], "k--", lw=3, label="Target")
        if bdt_s is not None:
            ax.plot(WAVELENGTHS, bdt_s, color=c_dataset, linestyle=":", lw=2, label="Best Dataset")
        ax.plot(surr_wls, surr_s, linestyle="-", color=c_surr, lw=3, label="Surrogate")
        ax.plot(WAVELENGTHS, rcwa_s, linestyle="-", color=c_physics, lw=2.5, label="Torcwa Physics")
        dataset_str2 = f" | Dataset {metric_name}={best_abs_for_mat:.3f}" if bdt_s is not None else ""
        ax.set_title(f"{mat_name} (S-Pol)\nTorcwa {metric_name}={rcwa_val:.2f}{unit} | Surr {metric_name}={r['loss']:.4f}{dataset_str2}")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Structure cross-section
        ax = ax_row[2]
        mat_color = MATERIAL_COLORS.get(mat_name, plt.cm.tab10(trained_mat_names.index(mat_name) % 10))
        xs = np.linspace(0, rcwa_config_dict.get("grating_period", 1000), 128)
        ax.plot(xs, profile_np, "k-", lw=2)
        ax.fill_between(xs, 0, profile_np, color=mat_color, alpha=0.6)
        ax.set_title(f"Structure Cross-Section\nFilm Height={h_val:.0f}nm, Inc Ang={inc_ang:.1f}°",)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Height (nm)")
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Harmonics amplitudes & phases
        ax_h = ax_row[3]
        n_harm = (len(geo) - 2) // 2
        px_np = px.numpy()
        amps_geo = px_np[:, 0]
        phases_geo = px_np[:, 1]
        x_pos = np.arange(1, n_harm + 1)
        ax_h.bar(x_pos, amps_geo, color=c_amp, edgecolor="black")
        ax_h.set_ylabel("Amplitude (nm)", color=c_amp)
        ax_h.tick_params(axis='y', labelcolor=c_amp, labelsize=12)
        ax_h.tick_params(axis='x', labelsize=12)
        ax_h.set_xlabel("Harmonic index")
        ax_h.set_title("Harmonic Composition (Optimized)")
        ax_p2 = ax_h.twinx()
        ax_p2.plot(x_pos, phases_geo, 'o', color=c_phase, markersize=10, markeredgecolor="black")
        ax_p2.set_ylabel("Phase (rad)", color=c_phase)
        ax_p2.tick_params(axis='y', labelcolor=c_phase, labelsize=12)
        ax_p2.set_ylim(-0.5, 2 * np.pi + 0.5)
        
        # 5th Column: Best Dataset Harmonics
        ax_h2 = ax_row[4]
        best_geo_for_mat = best_dataset_geo.get(mat_name)
        if best_geo_for_mat is not None:
            n_harm_data = (len(best_geo_for_mat) - 2) // 2
            amps_data = best_geo_for_mat[0:2*n_harm_data:2].numpy()
            phases_data = best_geo_for_mat[1:2*n_harm_data:2].numpy()
            h_data = best_geo_for_mat[-2].item()
            inc_data = best_geo_for_mat[-1].item()
            
            x_pos_data = np.arange(1, n_harm_data + 1)
            ax_h2.bar(x_pos_data, amps_data, color=c_amp, edgecolor="black")
            ax_h2.set_ylabel("Amplitude (nm)", color=c_amp)
            ax_h2.tick_params(axis='y', labelcolor=c_amp)
            ax_h2.tick_params(axis='x', labelsize=12)
            ax_h2.set_xlabel("Harmonic index")
            ax_h2.set_title(f"Dataset Best\n(Film Height={h_data:.0f}nm, Inc Ang={inc_data:.0f}°)")
            
            ax_p3 = ax_h2.twinx()
            ax_p3.plot(x_pos_data, phases_data, 'o', color=c_phase, markersize=10, markeredgecolor="black")
            ax_p3.set_ylabel("Phase (rad)", color=c_phase)
            ax_p3.tick_params(axis='y', labelcolor=c_phase, labelsize=12)
            ax_p3.set_ylim(-0.5, 2 * np.pi + 0.5)
        else:
            ax_h2.axis('off')
    
    plt.tight_layout()
    fig.suptitle(f"Surrogate Optimization: {args.mode.capitalize()}{title_suffix}", y=1.02)
        
    fig.tight_layout()
    
    out_path = out_dir / f"surrogate_optimization_{args.mode}.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    
    out_json = out_dir / f"surrogate_results_{args.mode}.json"
    with open(out_json, "w") as f:
        json.dump(metrics_list, f, indent=4)
    print(f"Saved metrics to {out_json}")

if __name__ == "__main__":
    main()
