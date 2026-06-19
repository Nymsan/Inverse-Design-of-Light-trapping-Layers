import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.checkpoint import get_best_forward_model, save_forward_checkpoint, FORWARD_MODEL_REGISTRY, load_forward_model
from Utils.surrogate_optimization import BatchedSurrogateOptimizer, recover_geometry_from_profile
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.models import GratingDataset

MATERIAL_LIBRARY = {
    "Si": "Si",
    "Si_Ag": "Si_Ag",
    "TiO2": "TiO2",
    "TiO2_Ag": "TiO2_Ag",
    "Si3N4": "Si3N4",
    "Si3N4_Ag": "Si3N4_Ag"
}

WAVELENGTHS = np.linspace(300, 1100, 161)

rcwa_config_dict = {
    'grating_period': 1000.0,
    'order_N': 15,
    'nx': 128,
    'height_per_layer': 5.0,
}

def get_random_bands():
    """Generate a random target band for active learning discovery."""
    if np.random.rand() > 0.5:
        return []  # Broadband
    else:
        # Narrow band
        c_s = np.random.uniform(400, 1000)
        w_s = np.random.uniform(100, 400)
        return [(c_s - w_s / 2, c_s + w_s / 2)]

def evaluate_oracle(geometries: torch.Tensor, mat_name: str, stats: dict, device: torch.device):
    """Evaluate a batch of geometries through the Torcwa Oracle."""
    n_samples = geometries.shape[0]
    n_harmonics = stats["n_harmonics"]
    n_fourier = n_harmonics * 2
    
    base_config = RCWAConfig(**rcwa_config_dict)
    if mat_name.endswith("_Ag"):
        base_config.grating_material = mat_name[:-3]
        base_config.reflector_type = 'Ag'
    else:
        base_config.grating_material = mat_name
        base_config.reflector_type = 'pec'
        
    wavelengths = torch.linspace(300, 1100, stats["n_wavelengths"] // 2, dtype=torch.float64, device=device)
    
    true_curves = []
    
    for i in tqdm(range(n_samples), desc=f"Torcwa Oracle [{mat_name}]"):
        px = geometries[i, :n_fourier].view(-1, 2).to(torch.float32)
        h_val = geometries[i, n_fourier].item()
        inc_ang = geometries[i, n_fourier+1].item()
        
        base_config.h = float(h_val)
        base_config.inc_ang = (float(inc_ang) + 1e-3) * np.pi/180.0
        base_config.azi_ang = 1e-3 * np.pi/180.0
        
        A_film, _ = get_absorptance_curve(
            params_x=px, params_y=None, wavelengths=wavelengths, config=base_config, show_progress=False
        )
        curve = torch.cat([A_film[:, 0], A_film[:, 1]], dim=0)
        true_curves.append(curve.cpu())
        
    return torch.stack(true_curves)

def load_al_dataset(mat_name: str, al_dir: Path):
    """Load previously accumulated Active Learning data for a material."""
    mat_file = al_dir / f"al_data_{mat_name}.pt"
    if mat_file.exists():
        data = torch.load(mat_file)
        return data["geometries"], data["curves"]
    return None, None

def save_al_dataset(geometries: torch.Tensor, curves: torch.Tensor, mat_name: str, al_dir: Path):
    """Append to and save Active Learning data for a material."""
    al_dir.mkdir(parents=True, exist_ok=True)
    mat_file = al_dir / f"al_data_{mat_name}.pt"
    
    if mat_file.exists():
        old_data = torch.load(mat_file)
        new_geos = torch.cat([old_data["geometries"], geometries], dim=0)
        new_curves = torch.cat([old_data["curves"], curves], dim=0)
    else:
        new_geos = geometries
        new_curves = curves
        
    torch.save({"geometries": new_geos, "curves": new_curves}, mat_file)
    print(f"Saved {len(geometries)} new AL samples. Total AL pool for {mat_name}: {len(new_geos)}")

def finetune_surrogate(model: nn.Module, mat_name: str, stats: dict, al_dir: Path, 
                       epochs: int = 5, lr: float = 1e-4, batch_size: int = 256):
    """Finetune the surrogate model on a mix of the AL dataset and a slice of the original dataset to prevent forgetting."""
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 1. Load AL Data
    al_geos, al_curves = load_al_dataset(mat_name, al_dir)
    if al_geos is None:
        return model
        
    # Normalize AL geos
    al_geos_norm = (al_geos - stats["geo_min"]) / (stats["geo_max"] - stats["geo_min"])
    al_geos_norm = torch.clamp(al_geos_norm, 0.0, 1.0)
    
    # Material index
    mat_idx = list(MATERIAL_LIBRARY.keys()).index(mat_name)
    al_mat_idx = torch.full((al_geos_norm.shape[0],), mat_idx, dtype=torch.long)
    al_dataset = TensorDataset(al_geos_norm, al_mat_idx, al_curves)
    
    # 2. Load Original Data (to prevent catastrophic forgetting)
    # We'll load just batch 0 and 1 (200 samples) as an anchor
    orig_dir = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_name}"
    orig_files = []
    for i in range(2):
        b_file = orig_dir / f"batch_{i:04d}.pt"
        if b_file.exists():
            orig_files.append(str(b_file))
            
    if orig_files:
        orig_ds = GratingDataset(
            {mat_name: orig_files}, 
            target_key="A_film_normal", 
            geo_min=stats["geo_min"], 
            geo_max=stats["geo_max"]
        )
        # orig_ds returns (geometry, material, target)
        orig_geos = orig_ds.geometry
        orig_curves = orig_ds.target
        orig_mat_idx = orig_ds.material_id
        orig_dataset = TensorDataset(orig_geos, orig_mat_idx, orig_curves)
        
        # Combine
        full_dataset = ConcatDataset([al_dataset, orig_dataset])
    else:
        full_dataset = al_dataset

    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nFinetuning Surrogate on {len(full_dataset)} samples ({len(al_geos)} AL + {len(full_dataset)-len(al_geos)} Orig) for {epochs} epochs...")
    
    model.train()
    for p in model.parameters():
        p.requires_grad = True
        
    pbar = tqdm(range(epochs), desc="Finetuning Surrogate", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        for g_b, m_b, c_b in dataloader:
            g_b, m_b, c_b = g_b.to(device), m_b.to(device), c_b.to(device)
            optimizer.zero_grad()
            preds = model(g_b, m_b)
            loss = criterion(preds, c_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(g_b)
        
        pbar.set_postfix(Loss=f"{epoch_loss/len(full_dataset):.4f}")
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Active Learning Loop for Surrogate Discovery")
    parser.add_argument('--iterations', type=int, default=3, help="Number of active learning loops")
    parser.add_argument('--proposals_per_mat', type=int, default=10, help="Structures proposed per material per iteration")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--ckpt_dir', type=str, default="Checkpoints/Si_TiO2_Si3N4", help="Directory containing the forward surrogate")
    parser.add_argument('--al_iter', type=int, default=-1, help="Active learning iteration of surrogate to use (-1 for latest, 0 for base)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    al_dir = PROJECT_ROOT / "Data" / "Active_Learning_Dataset"
    
    print("="*60)
    print("Starting Active Learning Loop")
    print(f"Iterations: {args.iterations} | Proposals per iteration per material: {args.proposals_per_mat}")
    print("="*60)
    
    # Load Initial Surrogate
    stats_path = ckpt_dir / "dataset_stats.pt"
    if not stats_path.exists():
        print("Error: dataset_stats.pt not found.")
        return
    stats = torch.load(stats_path, map_location="cpu")
    # Load model
    forward_model, fwd_name, _ = get_best_forward_model(
        ckpt_dir, 
        n_continuous=stats["n_continuous"], 
        n_wavelengths=stats["n_wavelengths"], 
        n_harmonics=stats["n_harmonics"],
        al_iter=args.al_iter
    )
    
    if forward_model is None:
        print("Error: No forward model found.")
        return
        
    forward_model.to(device)
    geo_min = stats["geo_min"].to(device)
    geo_max = stats["geo_max"].to(device)
    
    # Load original config so we can save new models cleanly
    original_ckpt_path = ckpt_dir / Path(fwd_name).name if "Active_Learning" not in fwd_name else ckpt_dir / fwd_name
    try:
        raw_ckpt = torch.load(original_ckpt_path, map_location="cpu", weights_only=False)
        model_config = raw_ckpt.get("model_config", {})
        model_class = raw_ckpt.get("model_class", Path(fwd_name).stem)
        history = raw_ckpt.get("history", {})
    except Exception as e:
        print(f"Failed to read model config from {original_ckpt_path}: {e}")
        model_config = {}
        model_class = "SkipCNN"
        history = {}
        
    al_iterations = history.get("al_iterations", 0)
    
    # Create output directories
    al_ckpt_dir = ckpt_dir / "Active_Learning"
    al_ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = al_ckpt_dir / "Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for it in range(args.iterations):
        al_iterations += 1
        print(f"\n>>> ACTIVE LEARNING ITERATION {it+1}/{args.iterations} (Global AL Iteration: {al_iterations}) <<<")
        
        for mat_idx, mat_name in enumerate(MATERIAL_LIBRARY.keys()):
            print(f"\n--- Material: {mat_name} ---")
            
            # 1. Propose candidates via Surrogate Optimization
            forward_model.eval()
            opt = BatchedSurrogateOptimizer(forward_model, geo_min, geo_max, device, nx=128)
            
            proposed_geos = []
            surr_losses = []
            
            for _ in range(args.proposals_per_mat):
                bands = get_random_bands()
                # Use fewer restarts to save time during AL inner loop
                opt_res = opt.optimize_geometry(bands, allowed_materials=[mat_idx], n_restarts=1000, steps=100, lr=0.1, top_k=1)
                for r in opt_res["top_results"]:
                    # AL dataset expects unnormalized flat geometry array (not a profile)
                    geo = r["geometry"] 
                    proposed_geos.append(geo)
                    surr_losses.append(r["loss"])
                    
            proposed_geos = torch.stack(proposed_geos).cpu()
            
            # 2. Oracle Verification (Torcwa)
            true_curves = evaluate_oracle(proposed_geos, mat_name, stats, device)
            
            # Compute MAE between what the surrogate predicted vs physical truth
            true_curves_gpu = true_curves.to(device)
            proposed_geos_gpu = proposed_geos.to(device)
            with torch.no_grad():
                surr_preds = forward_model(proposed_geos_gpu, torch.full((len(proposed_geos_gpu),), mat_idx, dtype=torch.long, device=device))
                mae_discrepancy = torch.nn.functional.l1_loss(surr_preds, true_curves_gpu).item()
                
            print(f"Oracle Verification Complete. Surrogate-Physics Discrepancy (MAE): {mae_discrepancy:.4f}")
            
            # --- PLOT DASHBOARD ---
            n_plots = len(proposed_geos)
            fig, axes = plt.subplots(n_plots, 2, figsize=(10, 3 * n_plots), squeeze=False, layout="constrained")
            wls = WAVELENGTHS
            for p_idx in range(n_plots):
                # P-pol
                ax_p = axes[p_idx, 0]
                ax_p.plot(wls, true_curves[p_idx][:len(wls)].numpy(), label="Torcwa", color="black", lw=2)
                ax_p.plot(wls, surr_preds[p_idx][:len(wls)].cpu().numpy(), label="Surrogate", color="red", linestyle="--", lw=2)
                ax_p.set_title(f"Proposal {p_idx+1} P-pol | MAE: {torch.nn.functional.l1_loss(surr_preds[p_idx], true_curves_gpu[p_idx]).item():.4f}")
                ax_p.grid(True, alpha=0.3)
                ax_p.set_ylabel("Absorptance")
                
                # S-pol
                ax_s = axes[p_idx, 1]
                ax_s.plot(wls, true_curves[p_idx][len(wls):].numpy(), label="Torcwa", color="black", lw=2)
                ax_s.plot(wls, surr_preds[p_idx][len(wls):].cpu().numpy(), label="Surrogate", color="red", linestyle="--", lw=2)
                ax_s.set_title(f"Proposal {p_idx+1} S-pol")
                ax_s.grid(True, alpha=0.3)
                
                if p_idx == n_plots - 1:
                    ax_p.set_xlabel("Wavelength (nm)")
                    ax_s.set_xlabel("Wavelength (nm)")
                if p_idx == 0:
                    ax_p.legend()
                    ax_s.legend()
                    
            import re
            base_stem = re.sub(r"_al\d+", "", Path(fwd_name).stem)
            fig.suptitle(f"Active Learning Discrepancy (Iter {al_iterations}) | {mat_name}", fontsize=16)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / f"al_discrepancy_{mat_name}_iter{al_iterations}.png")
            plt.close(fig)
            # ----------------------
            
            # 3. Save new data
            save_al_dataset(proposed_geos, true_curves, mat_name, al_dir)
            
            # 4. Finetune Surrogate
            forward_model = finetune_surrogate(forward_model, mat_name, stats, al_dir, epochs=5, lr=5e-5)
            
        # Save intermediate finetuned checkpoint globally
        history["val_loss"] = [mae_discrepancy] # arbitrary fallback
        history["al_iterations"] = al_iterations
        
        # strip any existing _alX suffix from fwd_name
        import re
        base_stem = re.sub(r"_al\d+", "", Path(fwd_name).stem)
        save_path = al_ckpt_dir / f"{base_stem}_al{al_iterations}.pt"
        
        save_forward_checkpoint(
            forward_model, 
            history, 
            str(save_path), 
            model_class, 
            model_config, 
            use_bfloat16=False
        )
        print(f"\n=> Saved Active Learning Checkpoint: {save_path.name}")

    print("\nActive Learning Loop Complete!")

if __name__ == "__main__":
    main()
