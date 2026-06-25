#!/usr/bin/env python
"""
Train SkipCNN3D on 3D grating datasets.

Usage:
    python Scripts/train_3d_forward.py \\
        --data_dir Data \\
        --materials Si TiO2 Si3N4 \\
        --epochs 500

Run consolidate_3d_dataset.py first to produce train_dataset.pt / val_dataset.pt
inside each LHS_3D_Dataset_<material> folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utils.models import _EMATracker          # reuse EMA from models.py
from Utils.models_3d import SkipCNN3D, Grating3DDataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_3d(
    model: SkipCNN3D,
    train_loader,
    val_loader,
    *,
    epochs: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 100,
    device,
    ema_decay: float = 0.999,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_patience = max(1, patience // 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-8
    )
    criterion = torch.nn.HuberLoss(delta=0.01, reduction="none")
    best_val, patience_ctr = float("inf"), 0
    ema = _EMATracker(model, decay=ema_decay)
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_max_err": []}

    from tqdm import tqdm
    pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep",
                dynamic_ncols=True, file=sys.stdout)

    for epoch in pbar:
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        train_accum = 0.0
        for batch in train_loader:
            px  = batch["params_x"].to(device)
            py  = batch["params_y"].to(device)
            h   = batch["h"].to(device)
            wl  = batch["wavelength"].to(device)
            tgt = batch["target"].to(device)         # (B, 2)  [p-pol, s-pol]

            pred = model(px, py, h, wl)              # (B, 2)

            # Weight loss by mean absorptance (reward high-performing structures)
            weights = tgt.mean(dim=-1, keepdim=True).clamp(min=0.1)
            loss = (criterion(pred, tgt) * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_accum += loss.item()

        avg_train = train_accum / len(train_loader)
        history["train_loss"].append(avg_train)

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        val_accum = val_mae = val_maxe = 0.0
        with torch.no_grad():
            for batch in val_loader:
                px  = batch["params_x"].to(device)
                py  = batch["params_y"].to(device)
                h   = batch["h"].to(device)
                wl  = batch["wavelength"].to(device)
                tgt = batch["target"].to(device)

                pred = model(px, py, h, wl)
                weights = tgt.mean(dim=-1, keepdim=True).clamp(min=0.1)
                loss = (criterion(pred, tgt) * weights).mean()

                val_accum += loss.item()
                abs_err = torch.abs(pred - tgt)
                val_mae  += abs_err.mean().item()
                val_maxe += abs_err.max(dim=1).values.mean().item()

        avg_val  = val_accum / len(val_loader)
        avg_mae  = val_mae   / len(val_loader)
        avg_maxe = val_maxe  / len(val_loader)
        history["val_loss"].append(avg_val)
        history["val_mae"].append(avg_mae)
        history["val_max_err"].append(avg_maxe)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            ema.snapshot()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                pbar.write(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

        pbar.set_postfix_str(
            f"lr={optimizer.param_groups[0]['lr']:.1e} "
            f"best={best_val:.3e} train={avg_train:.3e} val={avg_val:.3e} "
            f"vMAE={avg_mae:.4f} vMaxE={avg_maxe:.4f}"
        )

    ema.restore(model)
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train SkipCNN3D on 3D grating data")
    p.add_argument("--data_dir",  type=str, default="Data")
    p.add_argument("--materials", nargs="+", default=["Si", "TiO2", "Si3N4"])
    p.add_argument("--target_key", type=str, default="A_film_normal",
                   help="Which absorptance to predict: A_film_normal or A_grating_normal")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=100)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--num_workers",type=int,   default=4)
    # Architecture
    p.add_argument("--nx",              type=int,   default=64)
    p.add_argument("--conv_channels",   type=int, nargs="+", default=[8, 16, 16, 8])
    p.add_argument("--kernel_size",     type=int,   default=5)
    p.add_argument("--fc_dims",          type=int, nargs="+", default=[128])
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--grating_period",   type=float, default=1000.0)
    p.add_argument("--scalar_embed_dim", type=int,   default=16,
                   help="Projection size for the h and wavelength flat embeddings")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_root = PROJECT_ROOT / args.data_dir

    # ── load datasets ────────────────────────────────────────────────────────
    train_files, val_files = [], []
    for mat in args.materials:
        folder = data_root / f"LHS_3D_Dataset_{mat}"
        t = folder / "train_dataset.pt"
        v = folder / "val_dataset.pt"
        if not t.exists() or not v.exists():
            raise FileNotFoundError(
                f"Missing train/val datasets in {folder}. "
                "Run Scripts/consolidate_3d_dataset.py first."
            )
        train_files.append(str(t))
        val_files.append(str(v))

    train_set = Grating3DDataset(train_files, target_key=args.target_key)
    val_set   = Grating3DDataset(val_files,   target_key=args.target_key,
                                  h_min=train_set.h_min, h_max=train_set.h_max,
                                  wl_min=train_set.wl_min, wl_max=train_set.wl_max)

    print(f"Train: {len(train_set)} samples  |  Val: {len(val_set)} samples")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, pin_memory=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers
    )

    # ── model ────────────────────────────────────────────────────────────────
    model_kwargs = dict(
        n_harmonics=5,
        nx=args.nx,
        grating_period=args.grating_period,
        conv_channels=tuple(args.conv_channels),
        kernel_size=args.kernel_size,
        fc_dims=tuple(args.fc_dims),
        dropout=args.dropout,
        n_outputs=2,
        scalar_embed_dim=args.scalar_embed_dim,
    )
    model = SkipCNN3D(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SkipCNN3D  parameters: {n_params:,}")

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # ── checkpoint directory ─────────────────────────────────────────────────
    run_name = "3D_" + "_".join(args.materials)
    ckpt_dir = PROJECT_ROOT / "Checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset stats alongside checkpoint
    torch.save({
        "materials":    args.materials,
        "target_key":   args.target_key,
        "h_min":  train_set.h_min,
        "h_max":  train_set.h_max,
        "wl_min": train_set.wl_min,
        "wl_max": train_set.wl_max,
    }, ckpt_dir / "dataset_stats.pt")

    # ── train ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nTraining SkipCNN3D\n{'='*60}")
    t0 = time.time()
    history = train_3d(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed/60:.1f} min")

    # ── save ─────────────────────────────────────────────────────────────────
    ckpt_path = ckpt_dir / "skipcnn3d.pt"
    torch.save({
        "model_class":      "SkipCNN3D",
        "model_config":     model_kwargs,
        "model_state_dict": model.state_dict(),
        "history":          history,
    }, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    with open(ckpt_dir / "history_3d.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    best_val = min(history["val_loss"])
    best_mae = min(history["val_mae"])
    print(f"\nBest val loss : {best_val:.6e}")
    print(f"Best val MAE  : {best_mae:.6f}")


if __name__ == "__main__":
    main()
