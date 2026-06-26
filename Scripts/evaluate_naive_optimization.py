#!/usr/bin/env python
"""
Gradient-based naive optimization of grating parameters using RCWA + AdamW.

Matches the LHS_Dataset_Si simulation settings (order_N=20, 7 harmonics,
height_per_layer=5 nm, pec reflector) and uses differentiable parameter
transforms to keep amplitudes in [0, 15] nm and phases in [0, 2π].

Usage:
    # Pinned height at 2000 nm
    python Scripts/evaluate_naive_optimization.py --material Si --h_val 2000

    # Bounded height 1000-3000 nm
    python Scripts/evaluate_naive_optimization.py --material Si --h_val 1000 3000

    # Multiple restarts, custom iterations
    python Scripts/evaluate_naive_optimization.py --material TiO2 --h_val 2000 \\
        --n_iters 500 --n_restarts 5 --objective jsc
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.utils import (
    RCWAConfig, get_absorptance_curve, sun_weights, get_jsc_scaling_factor
)

# ---------------------------------------------------------------------------
# Simulation constants matching LHS_Dataset_Si
# ---------------------------------------------------------------------------
N_HARMONICS   = 7
ORDER_N       = 20          # same as LHS_Dataset_Si
NX            = 5000        # same as LHS_Dataset_Si
N_LAYERS      = 10          # same as LHS_Dataset_Si
HEIGHT_PER_LAYER = 5.0      # nm, same as LHS_Dataset_Si
GRATING_PERIOD   = 1000.0   # nm
REFLECTOR_TYPE   = 'pec'    # same as LHS_Dataset_Si
AMP_MAX       = 15.0        # nm — amplitude bound per harmonic
H_MIN, H_MAX  = 1000.0, 3000.0

WAVELENGTHS = torch.linspace(300, 1100, 161, dtype=torch.float32) + 1e-3  # stability offset

# Matplotlib styling
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

MATERIAL_COLORS = {
    "Si":    "#1f77b4",
    "TiO2":  "#ff7f0e",
    "Si3N4": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Parameter transforms
# ---------------------------------------------------------------------------

def transform_params(raw_params, raw_h=None, h_bounds=None):
    """
    Map unconstrained raw tensors to physical parameters.

    Amplitudes:  AMP_MAX * sigmoid(raw[:, 0])       -> [0, AMP_MAX] nm
    Phases:      2π      * sigmoid(raw[:, 1])        -> [0, 2π] rad
    Height (if free): H_MIN + (H_MAX - H_MIN) * sigmoid(raw_h)
                                                     -> [H_MIN, H_MAX] nm
    """
    amps   = AMP_MAX * torch.sigmoid(raw_params[:, 0])
    phases = 2.0 * np.pi * torch.sigmoid(raw_params[:, 1])
    params = torch.stack([amps, phases], dim=-1)

    if raw_h is not None and h_bounds is not None:
        lo, hi = h_bounds
        h = lo + (hi - lo) * torch.sigmoid(raw_h)
    else:
        h = None

    return params, h


def make_config(material, h_float, inc_ang=0.0):
    """Build an RCWAConfig with LHS_Dataset_Si settings."""
    eps = 1e-3
    return RCWAConfig(
        grating_period=GRATING_PERIOD,
        grating_period_y=None,          # 1D grating
        h=h_float,
        order_N=ORDER_N,
        order_N_y=None,
        nx=NX,
        ny=1,
        n_layers=N_LAYERS,
        height_per_layer=HEIGHT_PER_LAYER,
        subpixel=True,
        add_reflector=True,
        reflector_type=REFLECTOR_TYPE,
        inc_ang=(inc_ang + eps) * (np.pi / 180.0),
        azi_ang=eps * (np.pi / 180.0),
        grating_material=material,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def compute_loss(raw_params, raw_h, h_pinned, h_bounds, material, objective, device):
    """
    Returns scalar loss (to minimise) and the physical A_film curve.
    """
    params, h_tensor = transform_params(raw_params, raw_h, h_bounds)

    h_val = h_tensor.item() if h_tensor is not None else h_pinned
    config = make_config(material, h_val)

    wls = WAVELENGTHS.to(device)
    A_film, _ = get_absorptance_curve(
        params_x=params,
        params_y=None,
        wavelengths=wls,
        config=config,
        requires_grad=True,
    )

    if objective == 'jsc':
        S = sun_weights(wls)                          # [W/m²/nm]
        scale = get_jsc_scaling_factor(len(wls))
        jsc = scale * torch.sum(A_film * S * wls)
        loss = -jsc
    else:  # mean absorptance
        loss = -A_film.mean()

    return loss, A_film.detach()


# ---------------------------------------------------------------------------
# Single optimisation run
# ---------------------------------------------------------------------------

def run_optimization(material, h_pinned, h_bounds, objective, n_iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialise raw parameters near the middle of the bounded space
    # sigmoid(0) = 0.5  ->  amp = 0.5*AMP_MAX, phase = π
    raw_params = torch.zeros(N_HARMONICS, 2, dtype=torch.float32,
                             device=device, requires_grad=True)

    opt_vars = [raw_params]
    raw_h = None
    if h_bounds is not None:
        raw_h = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        opt_vars.append(raw_h)

    opt       = torch.optim.AdamW(opt_vars, lr=1e-1, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    best_loss   = float('inf')
    best_params = None
    best_h      = h_pinned
    loss_history = []

    pbar = tqdm(range(n_iters), desc=f"[seed={seed}]", leave=False)
    for it in pbar:
        opt.zero_grad()
        loss, A_film = compute_loss(raw_params, raw_h, h_pinned, h_bounds,
                                    material, objective, device)
        loss.backward()
        opt.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss   = loss_val
            best_params = raw_params.detach().clone()
            best_h      = raw_h.item() if raw_h is not None else h_pinned
            best_A_film = A_film.cpu().numpy()

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.4f}",
        })

    # Decode best params for reporting
    with torch.no_grad():
        phys_params, _ = transform_params(best_params, None, None)

    return {
        "best_loss":    best_loss,
        "best_h":       best_h,
        "best_params":  phys_params.cpu().numpy(),   # [N_HARMONICS, 2] (amp, phase)
        "best_A_film":  best_A_film,
        "loss_history": loss_history,
        "seed":         seed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results_list, material, h_pinned, h_bounds, objective,
                 out_dir: Path, label: str):
    color = MATERIAL_COLORS.get(material, "#1f77b4")
    wls_np = WAVELENGTHS.numpy()

    # Sort restarts by best loss
    results_list = sorted(results_list, key=lambda r: r["best_loss"])
    best = results_list[0]

    mode_str = f"{int(h_bounds[0])}-{int(h_bounds[1])}nm" if h_bounds else f"{int(h_pinned)}nm"
    h_display = f"{best['best_h']:.1f}" if h_bounds else f"{h_pinned:.0f}"

    # ---- Figure 1: Convergence curves ----
    fig_conv, ax = plt.subplots(figsize=(8, 5))
    for r in results_list:
        iters = range(len(r["loss_history"]))
        ax.plot(iters, [-v for v in r["loss_history"]],
                alpha=0.5, linewidth=1.2,
                label=f"seed={r['seed']} (best={-r['best_loss']:.4f})")
    ax.set_title(f"Optimisation Convergence\n({material}, h={mode_str}, obj={objective})")
    ax.set_xlabel("Iteration")
    ylabel = r"$J_{sc}$ (mA/cm$^2$)" if objective == 'jsc' else "Mean Absorptance"
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    conv_path = out_dir / f"convergence_{label}.png"
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f"Saved convergence plot: {conv_path}")

    # ---- Figure 2: Best absorptance spectrum ----
    fig_abs, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wls_np, best["best_A_film"], color=color, linewidth=1.8, label="Best $A_{film}$")

    # Shade Jsc integrand as background context
    from Utils.utils import sun_weights
    wls_t = torch.tensor(wls_np, dtype=torch.float32)
    S = sun_weights(wls_t).numpy()
    integrand = wls_np * S
    ax2 = ax.twinx()
    ax2.fill_between(wls_np, integrand / integrand.max(), alpha=0.08, color='goldenrod')
    ax2.set_ylabel(r"Norm. $\lambda \cdot I_{\mathrm{AM1.5g}}$ (a.u.)", color='goldenrod', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='goldenrod')
    ax2.set_ylim(0, 1)

    scale = get_jsc_scaling_factor(len(wls_np))
    jsc_val = scale * np.sum(best["best_A_film"] * S * wls_np)
    title_val = f"$J_{{sc}}$ = {jsc_val:.3f} mA/cm$^2$" if objective == 'jsc' \
                else f"Mean Abs = {best['best_A_film'].mean():.4f}"
    ax.set_title(f"Best Absorptance Spectrum ({material}, h={h_display} nm)\n{title_val}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Absorptance $A_{film}(\lambda)$")
    ax.set_xlim(300, 1100)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    spec_path = out_dir / f"best_spectrum_{label}.png"
    plt.savefig(spec_path, dpi=200)
    plt.close()
    print(f"Saved spectrum plot:     {spec_path}")

    return conv_path, spec_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Naive gradient-based RCWA optimisation (AdamW) — LHS_Dataset_Si settings.")
    p.add_argument("--material", type=str, default="Si",
                   choices=["Si", "TiO2", "Si3N4"],
                   help="Grating material.")
    p.add_argument("--h_val", type=float, nargs='+', default=[2000.0],
                   help="Film height. One value = pinned; two values = bounded range [lo, hi].")
    p.add_argument("--objective", type=str, default="jsc", choices=["jsc", "mean_abs"],
                   help="Optimisation objective: 'jsc' (default) or 'mean_abs'.")
    p.add_argument("--n_iters", type=int, default=500,
                   help="Number of AdamW iterations per restart.")
    p.add_argument("--n_restarts", type=int, default=3,
                   help="Number of independent random restarts.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (each restart uses seed + restart_index).")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory. Defaults to Results/naive_opt/<material>.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse height mode
    if len(args.h_val) == 1:
        h_pinned = args.h_val[0]
        h_bounds = None
        mode_label = f"pinned_{int(h_pinned)}nm"
    elif len(args.h_val) == 2:
        h_pinned = None
        h_bounds = (args.h_val[0], args.h_val[1])
        mode_label = f"bounded_{int(h_bounds[0])}-{int(h_bounds[1])}nm"
    else:
        raise ValueError("--h_val takes 1 (pinned) or 2 (range lo hi) values.")

    label = f"{args.material}_{mode_label}_{args.objective}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "Results" / "naive_opt" / args.material
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Naive RCWA Optimisation")
    print(f"  Material:   {args.material}")
    print(f"  Height:     {mode_label}")
    print(f"  Objective:  {args.objective}")
    print(f"  Harmonics:  {N_HARMONICS}  |  order_N: {ORDER_N}  |  nx: {NX}")
    print(f"  Amp bound:  0 – {AMP_MAX} nm per harmonic")
    print(f"  Restarts:   {args.n_restarts}  x  {args.n_iters} iterations")
    print(f"  Output:     {out_dir}")
    print("=" * 70)

    t0 = time.time()
    all_results = []
    for i in range(args.n_restarts):
        seed_i = args.seed + i
        print(f"\n--- Restart {i+1}/{args.n_restarts}  (seed={seed_i}) ---")
        res = run_optimization(
            material=args.material,
            h_pinned=h_pinned,
            h_bounds=h_bounds,
            objective=args.objective,
            n_iters=args.n_iters,
            device=device,
            seed=seed_i,
        )
        all_results.append(res)
        metric = -res["best_loss"]
        metric_name = "Jsc" if args.objective == "jsc" else "Mean Abs"
        print(f"  Best {metric_name}: {metric:.4f}  |  h = {res['best_h']:.1f} nm")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Pick global best
    best = min(all_results, key=lambda r: r["best_loss"])
    print("\n--- Best across all restarts ---")
    print(f"  Seed:    {best['seed']}")
    print(f"  h:       {best['best_h']:.2f} nm")
    metric_val = -best["best_loss"]
    metric_name = "Jsc (mA/cm²)" if args.objective == "jsc" else "Mean Abs"
    print(f"  {metric_name}: {metric_val:.4f}")
    print(f"  Amplitudes (nm): {best['best_params'][:, 0].round(3)}")
    print(f"  Phases (rad):    {best['best_params'][:, 1].round(3)}")

    # Save plots
    plot_results(all_results, args.material, h_pinned, h_bounds,
                 args.objective, out_dir, label)

    # Save results JSON
    results_summary = {
        "material":    args.material,
        "mode":        mode_label,
        "objective":   args.objective,
        "n_harmonics": N_HARMONICS,
        "order_N":     ORDER_N,
        "amp_max_nm":  AMP_MAX,
        "n_iters":     args.n_iters,
        "n_restarts":  args.n_restarts,
        "elapsed_s":   elapsed,
        "restarts": [
            {
                "seed":       r["seed"],
                "best_loss":  r["best_loss"],
                "best_h":     r["best_h"],
                "best_amps":  r["best_params"][:, 0].tolist(),
                "best_phases":r["best_params"][:, 1].tolist(),
            }
            for r in all_results
        ],
    }
    json_path = out_dir / f"results_{label}.json"
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results JSON:      {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
