import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import scipy.optimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.models import build_profile

# Default RCWA base configuration
rcwa_config_dict = {
    'grating_period': 1000.0,
    'order_N': 15,
    'nx': 128,
    'height_per_layer': 5.0,
}

MATERIAL_LIBRARY = {
    "Si": "Si",
    "Si_Ag": "Si_Ag",
    "TiO2": "TiO2",
    "TiO2_Ag": "TiO2_Ag",
    "Si3N4": "Si3N4",
    "Si3N4_Ag": "Si3N4_Ag"
}

def parse_bands(bands_args: list[float]) -> list[tuple[float, float]]:
    if not bands_args:
        return []
    if len(bands_args) % 2 != 0:
        raise ValueError("--bands must be pairs of floats")
    bands = []
    for i in range(0, len(bands_args), 2):
        bands.append((bands_args[i], bands_args[i+1]))
    return bands

def get_target_curve(wavelengths: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    if not bands:
        return np.ones(len(wavelengths) * 2)
    curve = np.zeros(len(wavelengths))
    for bmin, bmax in bands:
        mask = (wavelengths >= bmin) & (wavelengths <= bmax)
        curve[mask] = 1.0
    return np.concatenate([curve, curve])

# ---------------------------------------------------------------------------
# Objective Functions for SciPy
# ---------------------------------------------------------------------------
class TorcwaObjective:
    def __init__(self, target_curve: torch.Tensor, eval_wavelengths: torch.Tensor, 
                 mat_name: str, device: torch.device):
        self.target_curve = target_curve.to(device)
        self.eval_wavelengths = eval_wavelengths.to(device)
        self.mat_name = mat_name
        self.device = device
        
        # Setup base config
        self.base_config = RCWAConfig(**rcwa_config_dict)
        if mat_name.endswith("_Ag"):
            self.base_config.grating_material = mat_name[:-3]
            self.base_config.reflector_type = 'Ag'
        else:
            self.base_config.grating_material = mat_name
            self.base_config.reflector_type = 'pec'

        self.eval_count = 0
        self.best_loss = float('inf')
        self.best_x = None
        self.pbar = None
        
    def _vector_to_tensors(self, x: np.ndarray):
        """Converts flat 12D array to (h, inc_ang, px) tensors."""
        h = x[0]
        inc_ang = x[1]
        amps = x[2:7]
        phases = x[7:12]
        
        px_data = [[amps[j], phases[j]] for j in range(5)]
        px = torch.tensor(px_data, dtype=torch.float32, device=self.device)
        return h, inc_ang, px

    def evaluate(self, x: np.ndarray, requires_grad: bool = False):
        """Evaluates MAE loss for a given parameter vector."""
        if requires_grad:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)
            h = x_tensor[0]
            inc_ang = x_tensor[1]
            amps = x_tensor[2:7]
            phases = x_tensor[7:12]
            px = torch.stack([amps, phases], dim=1)
        else:
            h, inc_ang, px = self._vector_to_tensors(x)
            
        self.base_config.h = float(h)
        self.base_config.inc_ang = (float(inc_ang) + 1e-3) * np.pi / 180.0
        self.base_config.azi_ang = 1e-3 * np.pi / 180.0
        
        A_film, _ = get_absorptance_curve(
            params_x=px,
            params_y=None,
            wavelengths=self.eval_wavelengths,
            config=self.base_config,
            show_progress=False,
            requires_grad=requires_grad
        )
        
        # Concat P and S polarization
        sim_curve = torch.cat([A_film[:, 0], A_film[:, 1]], dim=0).to(self.device)
        
        loss = torch.nn.functional.mse_loss(sim_curve, self.target_curve)
        return loss, sim_curve

    def objective_no_grad(self, x: np.ndarray) -> float:
        """For Differential Evolution (no gradients)."""
        try:
            loss, _ = self.evaluate(x, requires_grad=False)
            val = loss.item()
        except Exception:
            return float('inf')
        
        self.eval_count += 1
        if val < self.best_loss:
            self.best_loss = val
            self.best_x = x.copy()
            if self.pbar:
                self.pbar.set_postfix({'best_mae': f"{val:.4f}"})
        if self.pbar:
            self.pbar.update(1)
            
        return val

# ---------------------------------------------------------------------------
# We will use SciPy's L-BFGS-B with finite difference approximation (`jac='2-point'`) 
# because RCWAConfig expects floats for `h` and `inc_ang`, making full 
# autograd through all parameters tricky without modifying Torcwa's internals.
# Finite difference is slower per step but guaranteed to be correct for all 12 params.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Naive Optimization Baseline (Direct Torcwa)")
    parser.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to optimize, e.g., --bands 500 750 800 900")
    parser.add_argument('--method', type=str, choices=['de', 'lbfgs', 'de_lbfgs'], default='de_lbfgs', 
                        help="Optimizer: Differential Evolution, L-BFGS-B, or DE followed by L-BFGS-B")
    parser.add_argument('--max_evals', type=int, default=1000, help="Max Torcwa evaluations per material")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--out_dir', type=str, default="Naive_Optimization")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = PROJECT_ROOT / "Results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        bands = parse_bands(args.bands)
    except ValueError as e:
        print(f"Error: {e}")
        return

    bands_str = "_".join([f"{int(b[0])}-{int(b[1])}" for b in bands]) if bands else "broadband"
    print(f"Target Bands: {bands if bands else 'Broadband (300-1100nm)'}")
    print(f"Method: {args.method.upper()}")

    # Add 1e-3 offset to avoid perfect symmetry/singular matrices in Torcwa
    WAVELENGTHS = np.linspace(300, 1100, 161) + 1e-3
    
    if bands:
        mask = np.zeros(len(WAVELENGTHS), dtype=bool)
        for bmin, bmax in bands:
            mask |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
        eval_wavelengths = WAVELENGTHS[mask]
    else:
        eval_wavelengths = WAVELENGTHS
        
    target_curve_np = np.ones(len(eval_wavelengths) * 2) # Target is 1.0 inside the bands
    target_curve = torch.tensor(target_curve_np, dtype=torch.float64, device=device)

    # Bounds for the 12 parameters
    # x = [h, inc_ang, a1..a5, p1..p5]
    bounds = [
        (500.0, 6000.0),   # h (nm)
        (0.0, 45.0),       # inc_ang (deg)
    ]
    bounds += [(0.0, 30.0)] * 5    # amps (nm)
    bounds += [(0.0, 2*np.pi)] * 5 # phases (rad)

    results_list = []

    for mat_name in MATERIAL_LIBRARY.keys():
        print(f"\n" + "="*50)
        print(f"Optimizing {mat_name}...")
        
        obj = TorcwaObjective(target_curve, torch.tensor(eval_wavelengths, dtype=torch.float64), mat_name, device)
        
        best_x = None
        best_loss = float('inf')
        
        # 1. Differential Evolution
        if 'de' in args.method:
            print("Running Differential Evolution...")
            obj.pbar = tqdm(total=args.max_evals, desc="DE Evals")
            
            # SciPy DE callback to stop if max_evals is reached
            def de_callback(xk, convergence):
                if obj.eval_count >= args.max_evals:
                    return True # Stop
                    
            # popsize=5 with 12 dims means 60 evals per generation
            res_de = scipy.optimize.differential_evolution(
                obj.objective_no_grad, 
                bounds=bounds, 
                maxiter=1000,
                popsize=5, 
                callback=de_callback,
                seed=42
            )
            obj.pbar.close()
            print(f"DE Finished. Best MAE: {res_de.fun:.4f} after {obj.eval_count} evals")
            best_x = res_de.x
            best_loss = res_de.fun
            
        # 2. L-BFGS-B
        if 'lbfgs' in args.method:
            print("Running L-BFGS-B (Finite Differences)...")
            
            # Initial guess: if DE ran, use its result, else use middle of bounds
            x0 = best_x if best_x is not None else np.array([(b[0]+b[1])/2 for b in bounds])
            
            evals_left = args.max_evals - obj.eval_count
            if evals_left > 0:
                obj.pbar = tqdm(total=evals_left, desc="L-BFGS Evals")
                
                res_lbfgs = scipy.optimize.minimize(
                    obj.objective_no_grad,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    jac='2-point', # Finite differences for gradients
                    options={'maxfun': evals_left, 'ftol': 1e-6}
                )
                obj.pbar.close()
                print(f"L-BFGS-B Finished. Best MAE: {res_lbfgs.fun:.4f}")
                best_x = res_lbfgs.x
                best_loss = res_lbfgs.fun
            else:
                print("Budget exhausted, skipping L-BFGS-B.")

        # Final evaluation on FULL wavelengths to get the complete curve for plotting
        obj.eval_wavelengths = torch.tensor(WAVELENGTHS, dtype=torch.float64, device=device)
        _, sim_curve = obj.evaluate(best_x)
        rcwa_p = sim_curve[:len(WAVELENGTHS)].cpu().numpy()
        rcwa_s = sim_curve[len(WAVELENGTHS):].cpu().numpy()
        
        # Calculate in-band average absorptance
        if bands:
            mask = np.zeros(len(WAVELENGTHS), dtype=bool)
            for bmin, bmax in bands:
                mask |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
        else:
            mask = np.ones(len(WAVELENGTHS), dtype=bool)
            
        rcwa_avg_abs = float((np.mean(rcwa_p[mask]) + np.mean(rcwa_s[mask])) / 2.0)
        
        # Save results
        h_val, inc_ang_val, px = obj._vector_to_tensors(best_x)
        geo_tensor = torch.cat([px.view(-1), torch.tensor([h_val, inc_ang_val], dtype=torch.float32, device=device)])
        
        results_list.append({
            "material": mat_name,
            "mae_loss": best_loss,
            "rcwa_avg_abs": rcwa_avg_abs,
            "h": float(h_val),
            "inc_ang": float(inc_ang_val),
            "geometry": geo_tensor.cpu().tolist(),
            "curve_p": rcwa_p.tolist(),
            "curve_s": rcwa_s.tolist()
        })

    # -----------------------------------------------------------------------
    # Plotting (Dashboard identical to evaluate_surrogate_optimization)
    # -----------------------------------------------------------------------
    n_results = len(results_list)
    fig, axes = plt.subplots(n_results, 4, figsize=(24, 6 * n_results), layout="constrained")
    if n_results == 1:
        axes = np.expand_dims(axes, axis=0)
        
    cmap = plt.cm.viridis
    c_physics = cmap(0.8)
    
    for idx, r in enumerate(results_list):
        ax_row = axes[idx]
        mat_name = r["material"]
        geo_t = torch.tensor(r["geometry"])
        
        # Plotting uses a padded array to visualize the target visually
        plot_target = np.zeros(len(WAVELENGTHS))
        if bands:
            for bmin, bmax in bands:
                plot_target[(WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)] = 1.0
        else:
            plot_target = np.ones(len(WAVELENGTHS))
            
        # P-Pol
        ax = ax_row[0]
        ax.plot(WAVELENGTHS, plot_target, "k-", lw=2, label="Target")
        ax.plot(WAVELENGTHS, r["curve_p"], linestyle="-", color=c_physics, lw=2, label="Torcwa")
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.2)
        ax.set_title(f"{mat_name} (P-Pol)\nTorcwa In-Band Abs={r['rcwa_avg_abs']:.3f} | MAE={r['mae_loss']:.4f}", fontsize=13)
        ax.set_xlim(300, 1100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavelength (nm) — P-Pol")
        ax.set_ylabel("Absorptance")
        if idx == 0: ax.legend(fontsize=9)
        
        # S-Pol
        ax = ax_row[1]
        ax.plot(WAVELENGTHS, plot_target, "k-", lw=2, label="Target")
        ax.plot(WAVELENGTHS, r["curve_s"], linestyle="-", color=c_physics, lw=2, label="Torcwa")
        if bands:
            for bmin, bmax in bands:
                ax.axvspan(bmin, bmax, color="gray", alpha=0.2)
        ax.set_title(f"{mat_name} (S-Pol)\nTorcwa In-Band Abs={r['rcwa_avg_abs']:.3f} | MAE={r['mae_loss']:.4f}", fontsize=13)
        ax.set_xlim(300, 1100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavelength (nm) — S-Pol")
        ax.set_ylabel("Absorptance")
        
        # Structure Profile
        ax = ax_row[2]
        prof_tensor, _, _ = build_profile(geo_t.unsqueeze(0), 5, nx=128)
        profile_np = prof_tensor[0].numpy()
        xs = np.linspace(0, 1000, 128)
        ax.plot(xs, profile_np, "k-", lw=1.5)
        ax.fill_between(xs, 0, profile_np, color=cmap(0.7), alpha=0.5)
        ax.set_title(f"Structure Cross-Section\nHeight={r['h']:.0f}nm, Inc={r['inc_ang']:.1f}°", fontsize=13)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Height (nm)")
        
        # Harmonics
        ax_h = ax_row[3]
        amps_geo = geo_t[:5].numpy()
        phases_geo = geo_t[5:10].numpy()
        x_pos = np.arange(1, 6)
        c_amp = cmap(0.3)
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

    out_path = out_dir / f"naive_optimization_{args.method}_{bands_str}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved dashboard to {out_path}")
    
    metrics_path = out_dir / f"naive_optimization_{args.method}_{bands_str}.json"
    with open(metrics_path, "w") as f:
        json.dump(results_list, f, indent=2)

if __name__ == "__main__":
    main()
