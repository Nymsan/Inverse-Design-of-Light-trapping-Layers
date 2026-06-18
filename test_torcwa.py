import torch
import numpy as np
from Utils.utils import get_absorptance_curve, RCWAConfig

WAVELENGTHS = np.linspace(300, 1100, 161)
px = torch.rand(20, 2)
config = RCWAConfig(grating_period=1000.0, h=100.0, order_N=15, nx=5000, ny=1, n_layers=10, height_per_layer=5.0, subpixel=True, add_reflector=True, reflector_type='pec', grating_material='Si')

print("Starting simulation...")
import time
start = time.time()
A_film, _ = get_absorptance_curve(params_x=px, params_y=None, wavelengths=torch.from_numpy(WAVELENGTHS).double(), config=config, show_progress=True)
print(f"Finished in {time.time() - start} seconds")
