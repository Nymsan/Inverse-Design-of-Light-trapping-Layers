#!/usr/bin/env python
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_comsol(h_nm: int):
    # COMSOL txt format: lambda0(nm), Probe 1 (grating), Probe 2 (film)
    data_dir = PROJECT_ROOT / "Data" / "AbsorptanceCurves"
    
    ppol_file = data_dir / f"Absorptance_curve_with_IBC_{h_nm}_nm_ppol.txt"
    spol_file = data_dir / f"Absorptance_curve_with_IBC_{h_nm}_nm_spol.txt"
    
    if not ppol_file.exists() or not spol_file.exists():
        print(f"COMSOL files not found for {h_nm}nm")
        return None, None, None
        
    ppol_data = np.loadtxt(ppol_file, comments="%")
    spol_data = np.loadtxt(spol_file, comments="%")
    
    wls = ppol_data[:, 0]
    # Use only Probe 2 for film absorptance
    abs_p = ppol_data[:, 2]
    abs_s = spol_data[:, 2]
    
    return wls, abs_p, abs_s

def interp_torcwa_to_comsol(torcwa_abs, comsol_wls):
    # Torcwa eval is usually 1601 points from 300 to 1100 nm
    # A_film is shape (1601, 2). 0 is p-pol, 1 is s-pol.
    torcwa_wls = np.linspace(300, 1100, len(torcwa_abs))
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
    # 1D Convergence (num_layers and order_N) against COMSOL
    c_wls, c_abs_p, c_abs_s = load_comsol(h_nm)
    if c_wls is None: return
    
    # --- NUM LAYERS ---
    file_layers = get_latest_file(f"sweep_num_layers{suffix}_{h_nm}nm_*.pt")
    if file_layers:
        data = torch.load(file_layers, map_location="cpu", weights_only=False)["results"]
        
        layers = []
        errors = []
        curves = {}
        fixed_ox = None
        
        for k, v in data.items():
            ox, oy, nl = extract_params(k)
            fixed_ox = ox
            t_abs = v["A_film"].numpy()
            t_p, t_s = interp_torcwa_to_comsol(t_abs, c_wls)
            err_p = np.mean(np.abs(t_p - c_abs_p))
            err_s = np.mean(np.abs(t_s - c_abs_s))
            
            layers.append(nl)
            errors.append((err_p, err_s))
            curves[nl] = (t_p, t_s)
            
        # Sort
        sort_idx = np.argsort(layers)
        layers = np.array(layers)[sort_idx]
        errors = [errors[i] for i in sort_idx]
        err_p_arr = np.array([e[0] for e in errors])
        err_s_arr = np.array([e[1] for e in errors])
        
        # Select 4 representative layer values (two near low, two near high)
        if len(layers) >= 4:
            idx_list = [0, 1, len(layers)-2, len(layers)-1]
            sel_layers = [layers[i] for i in idx_list]
        else:
            sel_layers = layers
            
        n_cols = len(sel_layers) + 1
        fig = plt.figure(figsize=(4 + 4*len(sel_layers), 8))
        gs = fig.add_gridspec(2, n_cols)
        
        # Error plot spans both rows in first column
        ax_err = fig.add_subplot(gs[:, 0])
        ax_err.plot(layers, err_p_arr, 'b^-', lw=2, label="p-pol")
        ax_err.plot(layers, err_s_arr, 'ro-', lw=2, label="s-pol")
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Number of Layers')
        ax_err.set_ylabel('Mean Absolute Error vs COMSOL')
        ax_err.set_title(f'Convergence')
        ax_err.legend()
        ax_err.grid(True, which="both", ls="-", alpha=0.5)
        
        # Spectra plots
        ax_p_ref, ax_s_ref = None, None
        for i, nl in enumerate(sel_layers):
            ax_p = fig.add_subplot(gs[0, i+1], sharey=ax_p_ref)
            if ax_p_ref is None: ax_p_ref = ax_p
            ax_s = fig.add_subplot(gs[1, i+1], sharey=ax_s_ref)
            if ax_s_ref is None: ax_s_ref = ax_s
            
            t_p, t_s = curves[nl]
            
            ax_p.plot(c_wls, c_abs_p, 'k--', lw=2, label="COMSOL")
            ax_p.plot(c_wls, t_p, alpha=0.8, label=f"Torcwa")
            ax_p.set_title(f"p-pol ({nl} layers)")
            if i == 0: ax_p.set_ylabel("Absorptance")
            else: plt.setp(ax_p.get_yticklabels(), visible=False)
            ax_p.legend()
            
            ax_s.plot(c_wls, c_abs_s, 'k--', lw=2, label="COMSOL")
            ax_s.plot(c_wls, t_s, alpha=0.8, label=f"Torcwa")
            ax_s.set_title(f"s-pol ({nl} layers)")
            ax_s.set_xlabel('Wavelength (nm)')
            if i == 0: ax_s.set_ylabel("Absorptance")
            else: plt.setp(ax_s.get_yticklabels(), visible=False)
            ax_s.legend()
        
        out_path = PROJECT_ROOT / "Results" / f"convergence_num_layers{suffix}_{h_nm}nm.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

    # --- ORDER N ---
    file_order = get_latest_file(f"sweep_order_N{suffix}_{h_nm}nm_*.pt")
    if file_order:
        data = torch.load(file_order, map_location="cpu", weights_only=False)["results"]
        
        orders = []
        errors = []
        curves = {}
        fixed_nl = None
        
        for k, v in data.items():
            ox, oy, nl = extract_params(k)
            fixed_nl = nl
            t_abs = v["A_film"].numpy()
            t_p, t_s = interp_torcwa_to_comsol(t_abs, c_wls)
            err_p = np.mean(np.abs(t_p - c_abs_p))
            err_s = np.mean(np.abs(t_s - c_abs_s))
            
            orders.append(ox)
            errors.append((err_p, err_s))
            curves[ox] = (t_p, t_s)
            
        # Sort
        sort_idx = np.argsort(orders)
        orders = np.array(orders)[sort_idx]
        errors = [errors[i] for i in sort_idx]
        err_p_arr = np.array([e[0] for e in errors])
        err_s_arr = np.array([e[1] for e in errors])
        
        # Select 4 representative order values (two near low, two near high)
        if len(orders) >= 4:
            idx_list = [0, 1, len(orders)-2, len(orders)-1]
            sel_orders = [orders[i] for i in idx_list]
        else:
            sel_orders = orders
            
        n_cols = len(sel_orders) + 1
        fig = plt.figure(figsize=(4 + 4*len(sel_orders), 8))
        gs = fig.add_gridspec(2, n_cols)
        
        # Error plot spans both rows in first column
        ax_err = fig.add_subplot(gs[:, 0])
        ax_err.plot(orders, err_p_arr, 'b^-', lw=2, label="p-pol")
        ax_err.plot(orders, err_s_arr, 'ro-', lw=2, label="s-pol")
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Fourier Order N')
        ax_err.set_ylabel('Mean Absolute Error vs COMSOL')
        ax_err.set_title(f'Convergence')
        ax_err.legend()
        ax_err.grid(True, which="both", ls="-", alpha=0.5)
        
        # Spectra plots
        ax_p_ref, ax_s_ref = None, None
        for i, ox in enumerate(sel_orders):
            ax_p = fig.add_subplot(gs[0, i+1], sharey=ax_p_ref)
            if ax_p_ref is None: ax_p_ref = ax_p
            ax_s = fig.add_subplot(gs[1, i+1], sharey=ax_s_ref)
            if ax_s_ref is None: ax_s_ref = ax_s
            
            t_p, t_s = curves[ox]
            
            ax_p.plot(c_wls, c_abs_p, 'k--', lw=2, label="COMSOL")
            ax_p.plot(c_wls, t_p, alpha=0.8, label=f"Torcwa")
            ax_p.set_title(f"p-pol (Order {ox})")
            if i == 0: ax_p.set_ylabel("Absorptance")
            else: plt.setp(ax_p.get_yticklabels(), visible=False)
            ax_p.legend()
            
            ax_s.plot(c_wls, c_abs_s, 'k--', lw=2, label="COMSOL")
            ax_s.plot(c_wls, t_s, alpha=0.8, label=f"Torcwa")
            ax_s.set_title(f"s-pol (Order {ox})")
            ax_s.set_xlabel('Wavelength (nm)')
            if i == 0: ax_s.set_ylabel("Absorptance")
            else: plt.setp(ax_s.get_yticklabels(), visible=False)
            ax_s.legend()
        
        out_path = PROJECT_ROOT / "Results" / f"convergence_order_N{suffix}_{h_nm}nm.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_combinations(h_nm: int, pattern_base: str):
    # For combinations, we compare against the highest fidelity run
    file_3d = get_latest_file(f"{pattern_base}_{h_nm}nm_*.pt")
    if not file_3d: return
    
    data = torch.load(file_3d, map_location="cpu", weights_only=False)["results"]
    
    parsed_runs = []
    for k, v in data.items():
        ox, oy, nl = extract_params(k)
        if ox is None: continue
        t_abs = v["A_film"].numpy().mean(axis=1) # Avg pol
        parsed_runs.append((ox, nl, t_abs))
        
    if not parsed_runs: return
    
    # Find highest order and highest layer as ground truth
    max_ox = max([r[0] for r in parsed_runs])
    max_nl = max([r[1] for r in parsed_runs])
    
    gt_abs = None
    for ox, nl, t_abs in parsed_runs:
        if ox == max_ox and nl == max_nl:
            gt_abs = t_abs
            break
            
    if gt_abs is None:
        print(f"Could not find exact max target ({max_ox}, {max_nl}).")
        return
        
    # Group by order
    orders = sorted(list(set([r[0] for r in parsed_runs])))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for order in orders:
        layers_for_order = []
        errs_for_order = []
        for ox, nl, t_abs in parsed_runs:
            if ox == order:
                err = np.mean(np.abs(t_abs - gt_abs))
                layers_for_order.append(nl)
                errs_for_order.append(err)
                
        # Sort by layer
        sort_idx = np.argsort(layers_for_order)
        layers_for_order = np.array(layers_for_order)[sort_idx]
        errs_for_order = np.array(errs_for_order)[sort_idx]
        
        # Don't plot the error of the ground truth against itself as log(0) is undefined
        # Remove exactly zero errors
        valid_mask = errs_for_order > 0
        if valid_mask.any():
            ax.plot(layers_for_order[valid_mask], errs_for_order[valid_mask], 'o-', label=f"Order {order}")
        
    ax.set_yscale('log')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel(f'Mean Abs Error vs (Order {max_ox}, Layers {max_nl})')
    display_name = pattern_base.replace("sweep_combination_", "").replace("sweep_combination", "").replace("_", " ").title()
    if display_name.strip():
        ax.set_title(f'{display_name} Convergence (grating height = {h_nm} nm)')
    else:
        ax.set_title(f'Convergence (grating height = {h_nm} nm)')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    out_path = PROJECT_ROOT / "Results" / f"convergence_{pattern_base}_{h_nm}nm.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    for h in [100, 1000]:
        plot_1d_convergence(h, suffix="")
        plot_1d_convergence(h, suffix="_no_subpixel")
        
        plot_combinations(h, pattern_base="sweep_combination_3d")
        plot_combinations(h, pattern_base="sweep_combination")
        plot_combinations(h, pattern_base="sweep_combination_no_subpixel")
