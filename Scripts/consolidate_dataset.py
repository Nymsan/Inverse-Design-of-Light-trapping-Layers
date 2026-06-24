#!/usr/bin/env python
import os
import glob
import torch
from pathlib import Path
from tqdm import tqdm

import argparse

def consolidate_dataset(data_dir: str, val_split: float, seed: int):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    batch_files = sorted(list(data_path.glob("batch_*.pt")))
    if not batch_files:
        print(f"No batch_*.pt files found in {data_dir}.")
        return

    print(f"Consolidating {len(batch_files)} batches from {data_dir}...")
    
    # Load first batch to see keys
    first_batch = torch.load(batch_files[0], map_location="cpu", weights_only=False)
    keys_to_concat = [k for k, v in first_batch.items() if isinstance(v, torch.Tensor) and v.ndim > 0]
    metadata = first_batch.get("metadata", {})
    
    consolidated = {k: [] for k in keys_to_concat}
    
    for bf in tqdm(batch_files):
        data = torch.load(bf, map_location="cpu", weights_only=False)
        for k in keys_to_concat:
            if k in data:
                consolidated[k].append(data[k])
                
    final_data = {}
    for k in keys_to_concat:
        final_data[k] = torch.cat(consolidated[k], dim=0)
            
    # Filter out samples with absorptance >= 1.0
    valid_idx = torch.ones(final_data[keys_to_concat[0]].shape[0], dtype=torch.bool)
    for key in ["A_film_normal", "A_grating_normal", "A_film_oblique", "A_grating_oblique"]:
        if key in final_data:
            A = final_data[key].view(final_data[key].shape[0], -1)
            valid_idx &= (A.max(dim=1).values < 1.0)
            
    num_total = valid_idx.shape[0]
    num_valid = valid_idx.sum().item()
    print(f"Filtering removed {num_total - num_valid} invalid samples out of {num_total} (>= 1.0 absorptance)")
    
    for k in keys_to_concat:
        final_data[k] = final_data[k][valid_idx]
        
    print("Dataset Limits:")
    for k in ["params_x", "h", "inc_ang"]:
        if k in final_data:
            print(f"  {k}: min={final_data[k].min().item():.4f}, max={final_data[k].max().item():.4f}")

        
    # Perform deterministic train/val split
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_valid, generator=rng)
    
    n_val = max(1, int(num_valid * val_split))
    v_idx = indices[:n_val]
    t_idx = indices[n_val:]
    
    train_data = {k: final_data[k][t_idx] for k in keys_to_concat}
    train_data["metadata"] = metadata
    
    val_data = {k: final_data[k][v_idx] for k in keys_to_concat}
    val_data["metadata"] = metadata
    
    train_file = data_path / "train_dataset.pt"
    val_file = data_path / "val_dataset.pt"
    
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    
    print(f"Saved {train_file} ({len(t_idx)} samples)")
    print(f"Saved {val_file} ({len(v_idx)} samples)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent / "Data"
    
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("LHS_Dataset_"):
            if "Ag" in folder.name or "deep" in folder.name.lower():
                continue
            consolidate_dataset(str(folder), args.val_split, args.seed)
            
    # Optionally also consolidate 3D datasets if they exist
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("LHS_3D_Dataset_"):
            if "Ag" in folder.name or "deep" in folder.name.lower():
                continue
            consolidate_dataset(str(folder), args.val_split, args.seed)
