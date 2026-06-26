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
    
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / "Si_TiO2_Si3N4"
    if not ckpt_dir.exists():
        print(f"Checkpoint directory {ckpt_dir} not found.")
        return
        
    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    n_continuous = stats["n_continuous"]
    n_wavelengths = stats["n_wavelengths"]
    n_harmonics = stats["n_harmonics"]
    
    model_names = ["skip_cnn.pt", "siren.pt", "forward_mlp.pt"]
    models = {}
    for mn in model_names:
        model_path = ckpt_dir / mn
        if model_path.exists():
            m, _, _ = load_forward_model(model_path, n_continuous=n_continuous, n_wavelengths=n_wavelengths, n_harmonics=n_harmonics)
            m = m.to(device)
            m.eval()
            models[mn] = m
        else:
            print(f"Warning: Model {model_path} not found.")

    # Load 10 random samples across all materials
    dataset_path = PROJECT_ROOT / "Data" / "LHS_Dataset_Si" / "val_dataset.pt"
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} not found.")
        return
        
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    
    mat_name_for_config = stats["materials"][0]
    first_batch_file = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_name_for_config}" / "batch_0000.pt"
    rcwa_config_dict = torch.load(first_batch_file, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    
    num_samples = 10
    
    val_h = data["val_geometry"][..., -2]
    indices = torch.randperm(val_h.shape[0])[:num_samples]
    
    geo = data["val_geometry"][indices].to(device)
    params_x = data["val_params_x"][indices].to(device)
    h = geo[..., -2]
    inc_ang = geo[..., -1]
    mat_id_tensor = data["val_material"][indices].to(device)

    model_timings = {}
    for mn, model in models.items():
        print(f"\nBenchmarking {mn}...")
        with torch.no_grad():
            from Utils.models import SpatialCNN, SkipCNN
            if isinstance(model, (SpatialCNN, SkipCNN)):
                profile, h_t, inc_t = build_profile(geo, n_harmonics, nx=128)
                def get_kwargs(idx=None):
                    if idx is None:
                        return {"profile": profile, "h": h_t, "inc_ang": inc_t, "material_id": mat_id_tensor}
                    return {"profile": profile[idx:idx+1], "h": h_t[idx:idx+1], "inc_ang": inc_t[idx:idx+1], "material_id": mat_id_tensor[idx:idx+1]}
            else:
                def get_kwargs(idx=None):
                    if idx is None:
                        return {"geometry": geo, "material_id": mat_id_tensor}
                    return {"geometry": geo[idx:idx+1], "material_id": mat_id_tensor[idx:idx+1]}

            _ = model(**get_kwargs())
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for i in range(num_samples):
                pred = model(**get_kwargs(i))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            surrogate_time = time.time() - t0
            
            # Batch surrogate timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            pred_batch = model(**get_kwargs())
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            surrogate_batch_time = time.time() - t0
            
        print(f"{mn} (Sequential 10 samples): {surrogate_time:.4f} s ({surrogate_time/num_samples:.5f} s/sample)")
        print(f"{mn} (Batched 10 samples): {surrogate_batch_time:.4f} s")
        
        model_timings[mn] = {
            "surrogate_sequential_total_s": surrogate_time,
            "surrogate_sequential_per_sample_s": surrogate_time/num_samples,
            "surrogate_batched_total_s": surrogate_batch_time,
            "surrogate_batched_per_sample_s": surrogate_batch_time/num_samples,
        }
    
    
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
        
        m_idx = mat_id_tensor[i].item()
        m_name = list(MATERIAL_LIBRARY.keys())[list(MATERIAL_LIBRARY.values()).index(m_idx)]
        
        t0 = time.time()
        _ = get_absorptance_curve(
            px,
            m_name,
            wavelengths,
            torcwa_config
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torcwa_time += (time.time() - t0)
        
    print(f"\nTorcwa Model (Sequential 10 samples): {torcwa_time:.4f} s ({torcwa_time/num_samples:.4f} s/sample)")
    
    results = {
        "torcwa_sequential_total_s": torcwa_time,
        "torcwa_sequential_per_sample_s": torcwa_time/num_samples,
        "num_samples": num_samples,
        "models": {}
    }
    
    for mn, timings in model_timings.items():
        seq_speedup = torcwa_time / timings["surrogate_sequential_total_s"]
        batch_speedup = torcwa_time / timings["surrogate_batched_total_s"]
        
        timings["speedup_sequential"] = seq_speedup
        timings["speedup_batched_surrogate"] = batch_speedup
        results["models"][mn] = timings
        
        print(f"\nSpeedup for {mn} (Sequential): {seq_speedup:.1f}x")
        print(f"Speedup for {mn} (Batched Surrogate): {batch_speedup:.1f}x")
    
    out_path = PROJECT_ROOT / "Results" / "timing_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved timing results to {out_path}")

if __name__ == "__main__":
    main()
