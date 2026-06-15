import torch
import numpy as np
from pathlib import Path

data_dir = Path("Data")
keys_to_check = ["A_film_normal", "A_film_oblique"]
WAVELENGTHS = np.linspace(300, 1100, 161)

exploded_details = []

for mat in ["Si", "TiO2", "Si3N4"]:
    folder = data_dir / f"LHS_Dataset_{mat}"
    if not folder.exists(): continue
    for f in folder.glob("batch_*.pt"):
        data = torch.load(f, map_location="cpu", weights_only=False)
        batch_size = data["h"].shape[0]
        
        for i in range(batch_size):
            for key in keys_to_check:
                if key in data:
                    curve = data[key][i]
                    if (curve > 1).any():
                        bad_indices = torch.where(curve > 1)
                        # In the batch dict, the curve shape is (161, 3) or (161, 2)
                        # bad_indices[0] corresponds to the wavelength dimension
                        bad_wls = sorted(list(set(WAVELENGTHS[idx] for idx in bad_indices[0].numpy())))
                        
                        exploded_details.append({
                            "material": mat,
                            "max_absorptance": curve.max().item(),
                            "bad_wavelengths": bad_wls,
                            "file": f.name,
                            "key": key,
                            "h": data["h"][i].item(),
                            "inc_ang": data.get("inc_ang", torch.zeros_like(data["h"]))[i].item(),
                            "params_x": data["params_x"][i].tolist()
                        })

exploded_details.sort(key=lambda x: x["max_absorptance"], reverse=True)
for d in exploded_details:
    wls = ", ".join([f"{w:.1f}" for w in d['bad_wavelengths']])
    print(f"[{d['material']}] {d['key']} in {d['file']} (Max: {d['max_absorptance']:.4f})")
    print(f"  h: {d['h']:.2f} nm, inc_ang: {d['inc_ang']:.4f} deg")
    print(f"  params_x: {d['params_x']}")
    print(f"  Exploded Wavelengths (nm): {wls}\n")
