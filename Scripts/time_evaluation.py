#!/usr/bin/env python
import time
import torch
import numpy as np
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.models import build_profile, MATERIAL_LIBRARY
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.checkpoint import load_forward_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Surrogate Model
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / "Si_TiO2_Si3N4"
    if not ckpt_dir.exists():
        print(f"Checkpoint directory {ckpt_dir} not found.")
        return
        
    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    model_path = ckpt_dir / "skip_cnn.pt"
    if not model_path.exists():
        print(f"Model {model_path} not found.")
        return
        
    model, _, _ = load_forward_model(model_path, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_harmonics=n_harmonics)
    model = model.to(device)
    model.eval()
    
    # Load 10 random samples
    mat_name = "Si"
    mat_dir = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_name}"
    dataset_path = mat_dir / "full_dataset.pt"
    
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} not found.")
        return
        
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    rcwa_config_dict = data.get("metadata", {}).get("config", {})
    
    # Take 10 samples
    num_samples = 10
    indices = torch.randperm(data["h"].shape[0])[:num_samples]
    
    params_x = data["params_x"][indices].to(device)
    h = data["h"][indices].to(device)
    inc_ang = data["inc_ang"][indices].to(device)
    
    geo_parts = [params_x.view(num_samples, -1), h.unsqueeze(-1), inc_ang.unsqueeze(-1)]
    geo = torch.cat(geo_parts, dim=-1)
    
    # Surrogate Timing
    # Warmup
    with torch.no_grad():
        profile, h_t, inc_t = build_profile(geo, n_harmonics, nx=128)
        mat_id = torch.full((num_samples,), MATERIAL_LIBRARY[mat_name], dtype=torch.long, device=device)
        _ = model(profile=profile, h=h_t, inc_ang=inc_t, material_id=mat_id)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        t0 = time.time()
        for i in range(num_samples):
            pred = model(profile=profile[i:i+1], h=h_t[i:i+1], inc_ang=inc_t[i:i+1], material_id=mat_id[i:i+1])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        surrogate_time = time.time() - t0
        
        # Batch surrogate timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        pred_batch = model(profile=profile, h=h_t, inc_ang=inc_t, material_id=mat_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        surrogate_batch_time = time.time() - t0
        
    print(f"\nSurrogate Model (Sequential 10 samples): {surrogate_time:.4f} s ({surrogate_time/num_samples:.5f} s/sample)")
    print(f"Surrogate Model (Batched 10 samples): {surrogate_batch_time:.4f} s")
    
    # Torcwa Timing
    torcwa_config = RCWAConfig(**rcwa_config_dict)
    wavelengths = torch.linspace(300, 1100, n_wavelengths//2, device=device, dtype=torch.float64)
    
    torcwa_time = 0.0
    for i in range(num_samples):
        h_i = h[i].item()
        inc_ang_i = inc_ang[i].item() * np.pi / 180.0
        
        torcwa_config.h = h_i
        torcwa_config.inc_ang = inc_ang_i + 1e-3  # Avoid normal incidence singularity
        torcwa_config.azi_ang = 1e-3
        
        px = params_x[i]
        
        t0 = time.time()
        _ = get_absorptance_curve(
            px,
            None,
            wavelengths,
            torcwa_config
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torcwa_time += (time.time() - t0)
        
    print(f"\nTorcwa Model (Sequential 10 samples): {torcwa_time:.4f} s ({torcwa_time/num_samples:.4f} s/sample)")
    print(f"\nSpeedup (Sequential): {torcwa_time / surrogate_time:.1f}x")
    print(f"Speedup (Batched Surrogate): {torcwa_time / surrogate_batch_time:.1f}x")
    
    results = {
        "surrogate_sequential_total_s": surrogate_time,
        "surrogate_sequential_per_sample_s": surrogate_time/num_samples,
        "surrogate_batched_total_s": surrogate_batch_time,
        "surrogate_batched_per_sample_s": surrogate_batch_time/num_samples,
        "torcwa_sequential_total_s": torcwa_time,
        "torcwa_sequential_per_sample_s": torcwa_time/num_samples,
        "speedup_sequential": torcwa_time / surrogate_time,
        "speedup_batched_surrogate": torcwa_time / surrogate_batch_time,
        "num_samples": num_samples
    }
    
    out_path = PROJECT_ROOT / "Results" / "timing_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved timing results to {out_path}")

if __name__ == "__main__":
    main()
