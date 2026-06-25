import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers')
from Utils.utils import get_material_eps

wavs_nm = np.linspace(300, 1100, 500)

n_si, k_si = [], []
n_tio2, k_tio2 = [], []
n_si3n4, k_si3n4 = [], []

for w in wavs_nm:
    eps_si = get_material_eps('Si', w)
    n_k = torch.sqrt(eps_si).item()
    n_si.append(n_k.real)
    k_si.append(n_k.imag)
    
    eps_tio2 = get_material_eps('TiO2', w)
    n_k = torch.sqrt(eps_tio2).item()
    n_tio2.append(n_k.real)
    k_tio2.append(n_k.imag)
    
    eps_si3n4 = get_material_eps('Si3N4', w)
    n_k = torch.sqrt(eps_si3n4).item()
    n_si3n4.append(n_k.real)
    k_si3n4.append(n_k.imag)

colors = {'Si': 'tab:blue', 'TiO2': 'tab:orange', 'Si3N4': 'tab:green'}

plt.figure(figsize=(8, 5))
plt.plot(wavs_nm, n_si, color=colors['Si'], label='Si (n)', linestyle='-')
plt.plot(wavs_nm, k_si, color=colors['Si'], label='Si (k)', linestyle='--')

plt.plot(wavs_nm, n_tio2, color=colors['TiO2'], label='TiO$_2$ (n)', linestyle='-')
plt.plot(wavs_nm, k_tio2, color=colors['TiO2'], label='TiO$_2$ (k)', linestyle='--')

plt.plot(wavs_nm, n_si3n4, color=colors['Si3N4'], label='Si$_3$N$_4$ (n)', linestyle='-')
plt.plot(wavs_nm, k_si3n4, color=colors['Si3N4'], label='Si$_3$N$_4$ (k)', linestyle='--')

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Refractive Index / Extinction Coefficient', fontsize=12)
plt.title('Optical Properties of Light-Trapping Materials', fontsize=14)
plt.legend(fontsize=10, ncol=3)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

os.makedirs('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers/Report/figures', exist_ok=True)
plt.savefig('/home/an/Documents/Inverse-Design-of-Light-trapping-Layers/Report/figures/refractive_indices.pdf')
print("Plot saved to figures/refractive_indices.pdf")
