#!/usr/bin/env python
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 20
})

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_comsol(h_nm: int):
    comsol_p = PROJECT_ROOT / "Data" / "AbsorptanceCurves" / f"Absorptance_curve_with_IBC_{h_nm}_nm_ppol.txt"
    comsol_s = PROJECT_ROOT / "Data" / "AbsorptanceCurves" / f"Absorptance_curve_with_IBC_{h_nm}_nm_spol.txt"
    if not comsol_p.exists() or not comsol_s.exists():
        return None, None, None
        
    data_p = np.loadtxt(comsol_p, comments='%')
    data_s = np.loadtxt(comsol_s, comments='%')
    return data_p[:, 0], data_p[:, 2], data_s[:, 2]

def interp_torcwa_to_comsol(torcwa_abs, comsol_wls):
    # torcwa_abs shape: [N_wls, 2]
    torcwa_wls = np.linspace(300, 1100, torcwa_abs.shape[0])
    torcwa_p = torcwa_abs[:, 0]
    torcwa_s = torcwa_abs[:, 1]
    
    return np.interp(comsol_wls, torcwa_wls, torcwa_p), np.interp(comsol_wls, torcwa_wls, torcwa_s)

def get_latest_file(pattern: str):
    data_dir = PROJECT_ROOT / "Data" / "AbsorptanceCurves"
    files = sorted(list(data_dir.glob(pattern)))
    if files:
        return files[-1]
    return None

def extract_params(key: str):
    m = re.match(r"order_x_(\d+)_order_y_(\d+)_layers_(\d+)", key)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m2 = re.match(r"order_x_(\d+)_layers_(\d+)", key)
    if m2:
        return int(m2.group(1)), int(m2.group(1)), int(m2.group(2))
    return None, None, None

def plot_1d_convergence(h_nm: int, suffix: str = ""):
    c_wls, c_abs_p, c_abs_s = load_comsol(h_nm)
    if c_wls is None: return
    
    # --- NUM LAYERS ---
    file_layers = get_latest_file(f"sweep_num_layers{suffix}_{h_nm}nm_*.pt")
    if file_layers:
        data = torch.load(file_layers, map_location="cpu", weights_only=False)["results"]
        
        layers = []
        errors = []
        curves = {}
        
        for k, v in data.items():
            ox, oy, nl = extract_params(k)
            t_abs = v["A_film"].numpy()
            t_p, t_s = interp_torcwa_to_comsol(t_abs, c_wls)
            
            mean_err_p = np.mean(np.abs(t_p - c_abs_p))
            mean_err_s = np.mean(np.abs(t_s - c_abs_s))
            max_err_p = np.max(np.abs(t_p - c_abs_p))
            max_err_s = np.max(np.abs(t_s - c_abs_s))
            
            layers.append(nl)
            errors.append((mean_err_p, mean_err_s, max_err_p, max_err_s))
            curves[nl] = (t_p, t_s)
            
        sort_idx = np.argsort(layers)
        layers = np.array(layers)[sort_idx]
        errors = [errors[i] for i in sort_idx]
        mean_p_arr = np.array([e[0] for e in errors])
        mean_s_arr = np.array([e[1] for e in errors])
        max_p_arr = np.array([e[2] for e in errors])
        max_s_arr = np.array([e[3] for e in errors])
        
        # Plot Mean and Max Error side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].plot(layers, mean_p_arr, 'b^-', lw=2, label="p-pol")
        axs[0].plot(layers, mean_s_arr, 'ro-', lw=2, label="s-pol")
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Number of Layers')
        axs[0].set_ylabel('Mean Absolute Error')
        axs[0].set_title(f'Mean Error (h={h_nm} nm)')
        axs[0].legend()
        axs[0].grid(True, which="both", ls="-", alpha=0.5)
        
        axs[1].plot(layers, max_p_arr, 'b^-', lw=2, label="p-pol")
        axs[1].plot(layers, max_s_arr, 'ro-', lw=2, label="s-pol")
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Number of Layers')
        axs[1].set_ylabel('Max Absolute Error')
        axs[1].set_title(f'Max Error (h={h_nm} nm)')
        axs[1].grid(True, which="both", ls="-", alpha=0.5)
        
        out_path = PROJECT_ROOT / "Results" / f"convergence_num_layers{suffix}_{h_nm}nm.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")
        
        # Plot individual spectra
        sel_layers = layers
            
        for nl in sel_layers:
            t_p, t_s = curves[nl]
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            axs[0].plot(c_wls, c_abs_p, 'k--', lw=3, label="COMSOL")
            axs[0].plot(c_wls, t_p, 'b-', alpha=0.8, lw=2, label="Torcwa")
            axs[0].set_title(f"p-pol Absorptance ({nl} layers)")
            axs[0].set_ylabel("Absorptance")
            axs[0].set_ylim(-0.05, 1.05)
            axs[0].legend()
            
            axs[1].plot(c_wls, c_abs_s, 'k--', lw=3, label="COMSOL")
            axs[1].plot(c_wls, t_s, 'r-', alpha=0.8, lw=2, label="Torcwa")
            axs[1].set_title(f"s-pol Absorptance ({nl} layers)")
            axs[1].set_ylabel("Absorptance")
            axs[1].set_xlabel('Wavelength (nm)')
            axs[1].set_ylim(-0.05, 1.05)
            axs[1].legend()
            
            s_path = PROJECT_ROOT / "Results" / f"spectra_num_layers{suffix}_{h_nm}nm_L{nl}.png"
            fig.tight_layout()
            print("Saving " + str(s_path)) ; fig.savefig(s_path)
            plt.close(fig)

    # --- ORDER N ---
    file_order = get_latest_file(f"sweep_order_N{suffix}_{h_nm}nm_*.pt")
    if file_order:
        data = torch.load(file_order, map_location="cpu", weights_only=False)["results"]
        
        orders = []
        errors = []
        curves = {}
        
        for k, v in data.items():
            ox, oy, nl = extract_params(k)
            t_abs = v["A_film"].numpy()
            t_p, t_s = interp_torcwa_to_comsol(t_abs, c_wls)
            
            mean_err_p = np.mean(np.abs(t_p - c_abs_p))
            mean_err_s = np.mean(np.abs(t_s - c_abs_s))
            max_err_p = np.max(np.abs(t_p - c_abs_p))
            max_err_s = np.max(np.abs(t_s - c_abs_s))
            
            orders.append(ox)
            errors.append((mean_err_p, mean_err_s, max_err_p, max_err_s))
            curves[ox] = (t_p, t_s)
            
        sort_idx = np.argsort(orders)
        orders = np.array(orders)[sort_idx]
        errors = [errors[i] for i in sort_idx]
        mean_p_arr = np.array([e[0] for e in errors])
        mean_s_arr = np.array([e[1] for e in errors])
        max_p_arr = np.array([e[2] for e in errors])
        max_s_arr = np.array([e[3] for e in errors])
        
        # Plot Mean and Max Error side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].plot(orders, mean_p_arr, 'b^-', lw=2, label="p-pol")
        axs[0].plot(orders, mean_s_arr, 'ro-', lw=2, label="s-pol")
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Fourier Order N')
        axs[0].set_ylabel('Mean Absolute Error')
        axs[0].set_title(f'Mean Error (h={h_nm} nm)')
        axs[0].legend()
        axs[0].grid(True, which="both", ls="-", alpha=0.5)
        
        axs[1].plot(orders, max_p_arr, 'b^-', lw=2, label="p-pol")
        axs[1].plot(orders, max_s_arr, 'ro-', lw=2, label="s-pol")
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Fourier Order N')
        axs[1].set_ylabel('Max Absolute Error')
        axs[1].set_title(f'Max Error (h={h_nm} nm)')
        axs[1].grid(True, which="both", ls="-", alpha=0.5)
        
        out_path = PROJECT_ROOT / "Results" / f"convergence_order_N{suffix}_{h_nm}nm.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")
        
        # Plot individual spectra
        sel_orders = orders
            
        for ox in sel_orders:
            t_p, t_s = curves[ox]
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            axs[0].plot(c_wls, c_abs_p, 'k--', lw=3, label="COMSOL")
            axs[0].plot(c_wls, t_p, 'b-', alpha=0.8, lw=2, label="Torcwa")
            axs[0].set_title(f"p-pol Absorptance (Order {ox})")
            axs[0].set_ylabel("Absorptance")
            axs[0].set_ylim(-0.05, 1.05)
            axs[0].legend()
            
            axs[1].plot(c_wls, c_abs_s, 'k--', lw=3, label="COMSOL")
            axs[1].plot(c_wls, t_s, 'r-', alpha=0.8, lw=2, label="Torcwa")
            axs[1].set_title(f"s-pol Absorptance (Order {ox})")
            axs[1].set_ylabel("Absorptance")
            axs[1].set_xlabel('Wavelength (nm)')
            axs[1].set_ylim(-0.05, 1.05)
            axs[1].legend()
            
            s_path = PROJECT_ROOT / "Results" / f"spectra_order_N{suffix}_{h_nm}nm_O{ox}.png"
            fig.tight_layout()
            print("Saving " + str(s_path)) ; fig.savefig(s_path)
            plt.close(fig)

def plot_combinations(h_nm: int, pattern_base: str):
    file_3d = get_latest_file(f"{pattern_base}_{h_nm}nm_*.pt")
    if not file_3d: return
    
    data = torch.load(file_3d, map_location="cpu", weights_only=False)
    results = data["results"]
    
    best_ox = -1
    best_nl = -1
    for k in results.keys():
        ox, _, nl = extract_params(k)
        best_ox = max(best_ox, ox)
        best_nl = max(best_nl, nl)
        
    best_key = None
    for k in results.keys():
        ox, _, nl = extract_params(k)
        if ox == best_ox and nl == best_nl:
            best_key = k
            break
            
    if best_key is None: return
    best_abs = results[best_key]["A_film"].numpy()
    
    orders = []
    layers = []
    errors = []
    
    for k, v in results.items():
        ox, _, nl = extract_params(k)
        t_abs = v["A_film"].numpy()
        
        # Both are from torcwa, so wls should be exactly the same length and values
        err_avg = np.mean(np.abs(t_abs - best_abs))
        
        orders.append(ox)
        layers.append(nl)
        errors.append(err_avg)
        
    if not orders: return
    
    orders = np.array(orders)
    layers = np.array(layers)
    errors = np.array(errors)
    
    # Line plot (each curve is a constant layer count)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_layers = np.unique(layers)
    for nl in unique_layers:
        idx = layers == nl
        o = orders[idx]
        e = errors[idx]
        
        sort_idx = np.argsort(o)
        
        # Don't plot the error=0 point if it messes up log scale, but log scale usually just hides 0.
        valid = e > 0
        if np.any(valid):
            ax.plot(o[sort_idx][valid[sort_idx]], e[sort_idx][valid[sort_idx]], 'o-', lw=2, label=f"{nl} layers")
        
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Fourier Order N')
    ax.set_ylabel('Mean Absolute Error vs Finest Eval')
    
    title_prefix = "3D Grating" if "3d" in pattern_base else "2D Grating"
    ax.set_title(f'{title_prefix} Combinations Error (h={h_nm} nm)')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    out_path = PROJECT_ROOT / "Results" / f"{pattern_base}_{h_nm}nm.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    for h in [100, 1000]:
        plot_1d_convergence(h)
        plot_1d_convergence(h, suffix="_no_subpixel")
        
        plot_combinations(h, "sweep_combination")
        plot_combinations(h, "sweep_combination_no_subpixel")
        plot_combinations(h, "sweep_combination_3d")
        plot_combinations(h, "sweep_combination_3d_no_subpixel")
