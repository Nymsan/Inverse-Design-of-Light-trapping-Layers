import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_convergence(data_path, out_path):
    # 1. Load the master dictionary
    saved_data = torch.load(data_path, weights_only=False)
    results = saved_data['results']
    metadata = saved_data['metadata']

    # 2. Extract the variables you need for plotting
    wavelengths = metadata['wavelengths']
    inc_ang = metadata['inc_ang_deg'] * np.pi / 180
    azi_ang = metadata['azi_ang_deg'] * np.pi / 180

    # 3. Determine what is sweeping by parsing keys
    # Example key: "order_x_10_order_y_10_layers_5"
    orders = set()
    layers = set()
    key_map = {}
    
    for key in results.keys():
        parts = key.split('_')
        o_x = int(parts[2])
        n_layers = int(parts[-1])
        orders.add(o_x)
        layers.add(n_layers)
        
    if len(orders) > 1 and len(layers) == 1:
        sweep_var = 'Order N'
        sweep_values = sorted(list(orders))
        for key in results.keys():
            parts = key.split('_')
            o_x = int(parts[2])
            key_map[o_x] = key
    elif len(layers) > 1 and len(orders) == 1:
        sweep_var = 'Number of Layers'
        sweep_values = sorted(list(layers))
        for key in results.keys():
            parts = key.split('_')
            n_layers = int(parts[-1])
            key_map[n_layers] = key
    else:
        raise ValueError("Cannot determine which variable is sweeping!")

    best_val = sweep_values[-1]
    best_abs = results[key_map[best_val]]['A_film']

    # Pre-calculate mean errors
    mean_errors_p = []
    mean_errors_s = []
    plot_vals = []

    for val in sweep_values:
        if val == best_val:
            continue
        
        curr_abs = results[key_map[val]]['A_film']
        
        err_p = torch.mean(torch.abs((curr_abs[:, 0] - best_abs[:, 0]))).item()
        err_s = torch.mean(torch.abs((curr_abs[:, 1] - best_abs[:, 1]))).item()
        
        mean_errors_p.append(err_p)
        mean_errors_s.append(err_s)
        plot_vals.append(val)

    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    (ax1_p, ax1_s), (ax2_p, ax2_s) = axs

    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))

    for idx, val in enumerate(sweep_values):
        curr_abs = results[key_map[val]]['A_film']
        
        # Top Row: Plot spectra
        ax1_p.plot(wavelengths.cpu(), curr_abs[:, 0].cpu(), label=f'{sweep_var} {val}', color=colors[idx], linewidth=1.5)
        ax1_s.plot(wavelengths.cpu(), curr_abs[:, 1].cpu(), label=f'{sweep_var} {val}', color=colors[idx], linewidth=1.5)
        
    # Format Top Row (Absorptance Spectra)
    ax1_p.set_xlabel('Wavelength (nm)')
    ax1_p.set_ylabel('Absorptance')
    ax1_p.set_title(f'p-Polarization Absorptance\nIncident angle: {inc_ang*180/np.pi:.2f}°, Azimuthal: {azi_ang*180/np.pi:.2f}°')
    ax1_p.grid(True, linestyle='--', alpha=0.6)
    ax1_p.legend(title=sweep_var, loc='best', fontsize='small')

    ax1_s.set_xlabel('Wavelength (nm)')
    ax1_s.set_ylabel('Absorptance')
    ax1_s.set_title(f's-Polarization Absorptance\nIncident angle: {inc_ang*180/np.pi:.2f}°, Azimuthal: {azi_ang*180/np.pi:.2f}°')
    ax1_s.grid(True, linestyle='--', alpha=0.6)
    ax1_s.legend(title=sweep_var, loc='best', fontsize='small')

    # Format Bottom Row (Mean Error Convergence)
    ax2_p.plot(plot_vals, mean_errors_p, marker='o', linestyle='-', color='blue')
    ax2_p.set_xlabel(sweep_var)
    ax2_p.set_ylabel(f'Mean Error (vs {best_val})')
    ax2_p.set_title('p-Polarization Convergence Trend')
    ax2_p.grid(True, linestyle='--', alpha=0.6)
    ax2_p.set_yscale('log')
    ax2_p.set_xscale('log')

    ax2_s.plot(plot_vals, mean_errors_s, marker='s', linestyle='-', color='red')
    ax2_s.set_xlabel(sweep_var)
    ax2_s.set_ylabel(f'Mean Error (vs {best_val})')         
    ax2_s.set_title('s-Polarization Convergence Trend')
    ax2_s.grid(True, linestyle='--', alpha=0.6)
    ax2_s.set_yscale('log')
    ax2_s.set_xscale('log')

    fig.suptitle(f'{sweep_var} Convergence')
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Convergence Plots")
    parser.add_argument('data', type=str, help="Path to the .pt data file (e.g., ../Data/AbsorptanceCurves/sweep_num_layers_2026-06-03.pt)")
    parser.add_argument('--out', type=str, default=None, help="Path to save the plot. Defaults to saving next to the data file.")
    args = parser.parse_args()
    
    if args.out is None:
        base, ext = os.path.splitext(args.data)
        out_path = base + "_convergence.png"
    else:
        out_path = args.out
        
    plot_convergence(args.data, out_path)
