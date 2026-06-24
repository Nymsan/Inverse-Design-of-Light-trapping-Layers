#!/usr/bin/env python
import os
import glob
import torch
from pathlib import Path
from tqdm import tqdm

def consolidate_dataset(data_dir: str):
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
    
    # We want to concatenate all keys that are tensors with a batch dimension
    keys_to_concat = [k for k, v in first_batch.items() if isinstance(v, torch.Tensor) and v.ndim > 0]
    
    # Also save metadata
    metadata = first_batch.get("metadata", {})
    
    consolidated = {k: [] for k in keys_to_concat}
    
    for bf in tqdm(batch_files):
        data = torch.load(bf, map_location="cpu", weights_only=False)
        for k in keys_to_concat:
            if k in data:
                consolidated[k].append(data[k])
                
    final_data = {}
    for k in keys_to_concat:
        try:
            final_data[k] = torch.cat(consolidated[k], dim=0)
        except Exception as e:
            print(f"Failed to concatenate key {k}: {e}")
            
    final_data["metadata"] = metadata
    
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
    
    out_file = data_path / "full_dataset.pt"
    torch.save(final_data, out_file)
    print(f"Saved {out_file} with {final_data[keys_to_concat[0]].shape[0]} total samples.\n")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "Data"
    
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("LHS_Dataset_"):
            consolidate_dataset(str(folder))
            
    # Optionally also consolidate 3D datasets if they exist
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("LHS_3D_Dataset_"):
            consolidate_dataset(str(folder))
