import torch
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/an/Documents/Inverse-Design-of-Light-trapping-Layers")
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.utils import RCWAConfig, get_absorptance_curve

dataset_path = PROJECT_ROOT / "Data" / "LHS_Dataset_Si" / "train_dataset.pt"
d = torch.load(dataset_path, weights_only=False)
ds_config = d['metadata']['config']

# We'll use the config from the dataset, but override h and incident angle
ds_config['h'] = 3000.0
ds_config['inc_ang'] = 1e-3 * np.pi/180
ds_config['azi_ang'] = 1e-3 * np.pi/180

# Convert to RCWAConfig object
config = RCWAConfig(**ds_config)

# Flat layer: 0 grating amplitude
px_data = torch.zeros((5, 2))

wavelengths = torch.from_numpy(np.linspace(300, 1100, 161)).double()

print("Simulating flat 3000nm Si layer...")
A_film, _ = get_absorptance_curve(
    params_x=px_data,
    params_y=None,
    wavelengths=wavelengths,
    config=config,
    show_progress=False
)

bdt_p = A_film[:, 0].cpu().numpy()
bdt_s = A_film[:, 1].cpu().numpy()

avg_p = np.mean(bdt_p)
avg_s = np.mean(bdt_s)

print(f"Average P-pol: {avg_p:.4f}")
print(f"Average S-pol: {avg_s:.4f}")
print(f"Total Average: {(avg_p + avg_s)/2:.4f}")
