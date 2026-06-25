import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

sys.path.append('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers')
from Utils.utils import get_absorptance_curve, RCWAConfig, get_absorptance, get_continuous_boundary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_row_data(params_x):
    config = RCWAConfig(
        grating_material='TiO2',
        inc_ang=math.radians(30),
        h=500.0,
        nx=150, 
        order_N=10, 
        add_reflector=True,
        reflector_type='Ag',
        grating_period=1000.0
    )
    
    wavs = torch.linspace(300, 1100, 1000, device=device)
    A_film, A_grating = get_absorptance_curve(params_x, None, wavs, config, show_progress=True)
    A_avg = (A_film[:, 0] + A_film[:, 1]).cpu().numpy() / 2.0
    
    sim_info = get_absorptance(params_x, None, 495, config)
    sim = sim_info[0]
    
    x_plot = torch.linspace(0, config.grating_period, 150, device=device)
    z_wavy, grating_height = get_continuous_boundary(x_plot, params_x, config.grating_period)
    grating_height = float(grating_height)
    z_top = grating_height + config.h
    
    z_plot = torch.linspace(-100, z_top + 100, 200, device=device)
    
    sim.source_planewave(amplitude=[1, 0], direction='forward', notation='ps')
    [Ex_p, Ey_p, Ez_p], _ = sim.field_xz(x_plot, z_plot, y=torch.tensor(0.0, device=device))
    Enorm_p = torch.sqrt(torch.abs(Ex_p)**2 + torch.abs(Ey_p)**2 + torch.abs(Ez_p)**2)
    
    sim.source_planewave(amplitude=[0, 1], direction='forward', notation='ps')
    [Ex_s, Ey_s, Ez_s], _ = sim.field_xz(x_plot, z_plot, y=torch.tensor(0.0, device=device))
    Enorm_s = torch.sqrt(torch.abs(Ex_s)**2 + torch.abs(Ey_s)**2 + torch.abs(Ez_s)**2)
    
    Enorm_avg = (Enorm_p + Enorm_s).cpu().numpy() / 2.0
    
    return wavs.cpu().numpy(), A_avg, x_plot.cpu().numpy(), z_plot.cpu().numpy(), Enorm_avg, z_wavy.cpu().numpy(), grating_height, z_top, config

params_flat = torch.zeros((10, 2), device=device)
params_grating = torch.tensor([[10.0, 0.0], [12.0, 2.5], [14.0, 1.0]], device=device)

data_flat = get_row_data(params_flat)
data_grating = get_row_data(params_grating)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

for i, data in enumerate([data_flat, data_grating]):
    wavs, A_avg, x_plot, z_plot, Enorm_avg, z_wavy, grating_height, z_top, config = data
    
    # Col 0: Schematic
    ax = axs[i, 0]
    ax.fill_between(x_plot, z_top, z_top + 150, color='silver', label='Ag Backreflector')
    ax.fill_between(x_plot, grating_height, z_top, color='tab:blue', alpha=0.5, label='Si Film')
    ax.fill_between(x_plot, z_wavy, grating_height, color='tab:orange', alpha=0.5, label='TiO$_2$ Grating')
    ax.plot(x_plot, z_wavy, color='black', lw=2.0)
    ax.axhline(grating_height, color='black', lw=1.5)
    ax.axhline(z_top, color='black', lw=1.5)
    
    ax.set_ylim(z_top + 100, -100)
    ax.set_xlim(0, config.grating_period)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('z (nm)')
    if i == 0:
        ax.set_title('Flat Structure Profile')
        ax.legend(loc='lower left')
    else:
        ax.set_title('Grating Structure Profile')
        
    # Col 1: Absorptance
    ax = axs[i, 1]
    ax.plot(wavs, A_avg, color='purple', lw=2.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorptance (Avg pol)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    if i == 0:
        ax.set_title('Flat Film Absorptance')
    else:
        ax.set_title('Grating Absorptance')
        
    # Col 2: Field
    ax = axs[i, 2]
    vmax = np.max(Enorm_avg)
    im = ax.pcolormesh(x_plot, z_plot, Enorm_avg.T, shading='auto', cmap='magma', vmax=vmax, rasterized=True, linewidth=0, edgecolors='none')
    ax.plot(x_plot, z_wavy, color='w', linestyle='-', lw=2.0)
    ax.axhline(grating_height, color='w', linestyle='--', lw=1.5)
    ax.axhline(z_top, color='w', linestyle='--', lw=1.5)
    ax.set_ylim(z_top + 100, -100)
    ax.set_xlim(0, config.grating_period)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('z (nm)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    if i == 0:
        ax.set_title('Avg |E| field at 495 nm')
    else:
        ax.set_title('Avg |E| field at 495 nm')

plt.tight_layout()
os.makedirs('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers/Report/figures', exist_ok=True)
plt.savefig('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers/Report/figures/fabry_perot_grid.pdf', dpi=300)
print("Plot saved to figures/fabry_perot_grid.pdf")
