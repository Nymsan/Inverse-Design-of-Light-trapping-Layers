import torch
from pathlib import Path

data_dir = Path("Data")
for mat in ["Si", "TiO2", "Si3N4"]:
    folder = data_dir / f"LHS_Dataset_{mat}"
    files = list(folder.glob("batch_*.pt"))
    for f in files:
        data = torch.load(f, weights_only=False)
        for key in ["A_film_normal"]:
            if key in data:
                val = data[key]
                if (val < -1e-4).any():
                    min_val = val.min().item()
                    print(f"[{mat}] {f.name} {key} has min absorptance {min_val:.4f} < 0.0")
                    break
