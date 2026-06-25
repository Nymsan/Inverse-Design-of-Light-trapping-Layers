#!/usr/bin/env python
"""
Consolidate 3D LHS dataset batches (batch_*.pt) into train_dataset.pt / val_dataset.pt.

Each batch contains:
  wavelength, h, inc_ang, azi_ang,
  params_x (B, 5, 2), params_y (B, 5, 2),
  A_film_normal (B, 2), A_grating_normal (B, 2),
  A_film_max_wl (B, 2), A_grating_max_wl (B, 2)
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def consolidate_3d(data_dir: str, val_split: float = 0.20, seed: int = 42):
    data_path = Path(data_dir)
    batch_files = sorted(data_path.glob("batch_*.pt"))
    if not batch_files:
        print(f"No batch_*.pt files found in {data_dir}.")
        return

    print(f"Consolidating {len(batch_files)} batches from {data_path.name} ...")

    first = torch.load(batch_files[0], map_location="cpu", weights_only=False)
    keys = [k for k, v in first.items() if isinstance(v, torch.Tensor) and v.ndim > 0]
    metadata = first.get("metadata", {})

    buckets = {k: [] for k in keys}
    for bf in tqdm(batch_files):
        data = torch.load(bf, map_location="cpu", weights_only=False)
        for k in keys:
            if k in data:
                buckets[k].append(data[k])

    merged = {k: torch.cat(buckets[k], dim=0) for k in keys}

    # Filter samples with absorptance >= 1.0
    valid = torch.ones(merged[keys[0]].shape[0], dtype=torch.bool)
    for key in ("A_film_normal", "A_grating_normal", "A_film_max_wl", "A_grating_max_wl"):
        if key in merged:
            A = merged[key].view(merged[key].shape[0], -1)
            valid &= (A.max(dim=1).values < 1.0)

    n_total = valid.shape[0]
    n_valid = valid.sum().item()
    print(f"Filtered {n_total - n_valid} invalid samples (absorptance >= 1.0), keeping {n_valid}.")

    merged = {k: v[valid] for k, v in merged.items()}

    # Train / val split
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_valid, generator=rng)
    n_val = max(1, int(n_valid * val_split))
    v_idx, t_idx = perm[:n_val], perm[n_val:]

    train_data = {k: merged[k][t_idx] for k in keys}
    train_data["metadata"] = metadata
    val_data = {k: merged[k][v_idx] for k in keys}
    val_data["metadata"] = metadata

    torch.save(train_data, data_path / "train_dataset.pt")
    torch.save(val_data,   data_path / "val_dataset.pt")
    print(f"  train: {len(t_idx)} samples  →  {data_path / 'train_dataset.pt'}")
    print(f"  val:   {len(v_idx)} samples  →  {data_path / 'val_dataset.pt'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split", type=float, default=0.20)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent / "Data"
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("LHS_3D_Dataset_"):
            consolidate_3d(str(folder), args.val_split, args.seed)
