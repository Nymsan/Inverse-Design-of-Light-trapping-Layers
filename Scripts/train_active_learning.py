import os
import sys
import argparse
import time
import json
import glob
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.checkpoint import get_best_forward_model, save_forward_checkpoint, FORWARD_MODEL_REGISTRY, load_forward_model
from Utils.surrogate_optimization import BatchedSurrogateOptimizer, recover_geometry_from_profile
from Utils.utils import RCWAConfig, get_absorptance_curve
from Utils.models import GratingDataset, MATERIAL_LIBRARY

WAVELENGTHS = np.linspace(300, 1100, 161)



def get_random_bands():
    """Generate a random target band for active learning discovery."""
    if np.random.rand() > 0.5:
        return []  # Broadband
    else:
        # Narrow band
        c_s = np.random.uniform(400, 1000)
        w_s = np.random.uniform(100, 400)
        return [(c_s - w_s / 2, c_s + w_s / 2)]

def evaluate_oracle(geometries: torch.Tensor, mat_name: str, stats: dict, device: torch.device, order_N: int = None, height_per_layer: float = None):
    """Evaluate a batch of geometries through the Torcwa Oracle."""
    n_samples = geometries.shape[0]
    n_harmonics = stats["n_harmonics"]
    n_fourier = n_harmonics * 2

    # Extract config from first available dataset batch
    rcwa_config_dict = {}
    try:
        prefix = stats.get("dataset_prefixes", ["LHS_Dataset"])[0]
        mat_dir = PROJECT_ROOT / "Data" / f"{prefix}_{mat_name}"
        first_batch = mat_dir / "train_dataset.pt"
        rcwa_config_dict = torch.load(first_batch, map_location="cpu", weights_only=False).get("metadata", {}).get("config", {})
    except StopIteration:
        pass
    base_config = RCWAConfig(**rcwa_config_dict)
    if mat_name.endswith("_Ag"):
        base_config.grating_material = mat_name[:-3]
        base_config.reflector_type = 'Ag'
    else:
        base_config.grating_material = mat_name
        base_config.reflector_type = 'pec'
        
    if order_N is not None:
        base_config.order_N = order_N
    if height_per_layer is not None:
        base_config.height_per_layer = height_per_layer
        
    wavelengths = torch.linspace(300, 1100, stats["n_wavelengths"] // 2, dtype=torch.float64, device=device) + 1e-3
    
    true_curves = []
    
    for i in tqdm(range(n_samples), desc=f"Torcwa Oracle [{mat_name}]"):
        px = geometries[i, :n_fourier].view(-1, 2).to(torch.float32)
        h_val = geometries[i, n_fourier].item()
        inc_ang = geometries[i, n_fourier+1].item()
        
        base_config.h = float(h_val)
        base_config.inc_ang = (float(inc_ang) + 1e-3) * np.pi/180.0
        base_config.azi_ang = 1e-3 * np.pi/180.0
        
        A_film, _ = get_absorptance_curve(
            params_x=px, params_y=None, wavelengths=wavelengths, config=base_config, show_progress=True
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

def finetune_surrogate(model: nn.Module, trained_materials: list, stats: dict, al_dir: Path, 
                       epochs: int = 1000, lr: float = 1e-4, batch_size: int = 256, patience: int = 50):
    """Finetune the surrogate model on a mix of the AL dataset and a slice of the original dataset to prevent forgetting."""
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    all_geos_raw = []
    all_mat_idxs = []
    all_curves = []
    data_files_dict = {}
    
    for mat_name in trained_materials:
        # 1. Load AL Data
        al_geos, al_curves = load_al_dataset(mat_name, al_dir)
        mat_idx = list(MATERIAL_LIBRARY.keys()).index(mat_name)
        
        if al_geos is not None:
            al_mat_idx = torch.full((al_geos.shape[0],), mat_idx, dtype=torch.long)
            
            all_geos_raw.append(al_geos)
            all_mat_idxs.append(al_mat_idx)
            all_curves.append(al_curves)
            
        # 2. Load Original Data (to prevent catastrophic forgetting)
        orig_dir = PROJECT_ROOT / "Data" / f"LHS_Dataset_{mat_name}"
        train_file = orig_dir / "train_dataset.pt"
        if train_file.exists():
            data_files_dict[mat_name] = [str(train_file)]
            
    if len(all_geos_raw) == 0:
        return model
        
    al_dataset = TensorDataset(torch.cat(all_geos_raw), torch.cat(all_mat_idxs), torch.cat(all_curves))
        
    if data_files_dict:
        orig_ds = GratingDataset(
            data_files=data_files_dict, target_key="A_film_normal",
            geo_min=stats["geo_min"], geo_max=stats["geo_max"]
        )
        # orig_ds returns (geometry, material, target)
        
        # Subsample original dataset to ~2000 samples to prevent swamping the AL data
        num_orig = len(orig_ds)
        subset_size = min(2000, num_orig)
        indices = torch.randperm(num_orig)[:subset_size]
        
        orig_geos = orig_ds.geometry[indices]
        orig_curves = orig_ds.target[indices]
        orig_mat_idx = orig_ds.material_id[indices]
        orig_dataset = TensorDataset(orig_geos, orig_mat_idx, orig_curves)
        
        # Combine
        full_dataset = ConcatDataset([al_dataset, orig_dataset])
    else:
        full_dataset = al_dataset

    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    total_al_samples = sum(len(g) for g in all_geos_raw)
    total_orig_samples = len(orig_dataset) if data_files_dict else 0
    print(f"\nFinetuning Surrogate on {len(full_dataset)} samples ({total_al_samples} AL + {total_orig_samples} Orig) for up to {epochs} epochs...")
    
    model.train()
    for p in model.parameters():
        p.requires_grad = True
        
    patience_counter = 0
    best_loss = float('inf')
    
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
        
        avg_loss = epoch_loss / len(full_dataset)
        pbar.set_postfix(Loss=f"{avg_loss:.4f}")
        
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  -> Early stopping at epoch {epoch+1} (loss={best_loss:.4f})")
            break
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Active Learning Loop for Surrogate Discovery")
    parser.add_argument('--iterations', type=int, default=3, help="Number of active learning loops")
    parser.add_argument('--proposals_per_mat', type=int, default=10, help="Structures proposed per material per iteration")
    parser.add_argument('--mode', type=str, choices=["geometry", "de"], default="de", help="Optimization mode for generating proposals")
    parser.add_argument('--restarts', type=int, default=1000, help="Population size for DE or n_restarts for Gradient Ascent")
    parser.add_argument('--steps', type=int, default=100, help="Generations for DE or steps for Gradient Ascent")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--ckpt_dir', type=str, default="Checkpoints/Si_TiO2_Si3N4", help="Directory containing the forward surrogate")
    parser.add_argument('--al_iter', type=int, default=-1, help="Active learning iteration of surrogate to use (-1 for latest, 0 for base)")
    parser.add_argument('--force_forward_model', type=str, default=None, help="Force active learning to use a specific forward model (e.g. skip_cnn.pt)")
    parser.add_argument('--h_val', nargs="+", type=float, default=None, help="Constrain the active learning exploration to a specific height (nm) or range (min max)")
    parser.add_argument('--inc_val', type=float, default=None, help="Constrain the active learning exploration to a specific incident angle (degrees)")
    parser.add_argument('--expand_amps', type=float, default=None, help="Temporarily expand the maximum amplitude bounds (e.g., to 25.0 nm) for Active Learning")
    parser.add_argument('--order_N', type=int, default=None, help="RCWA Order N override for Torcwa evaluation")
    parser.add_argument('--height_per_layer', type=float, default=None, help="RCWA height per layer override for Torcwa evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    provided_path = Path(args.ckpt_dir)
    if provided_path.is_absolute() or provided_path.exists():
        ckpt_dir = provided_path.resolve()
    else:
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
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    # Load model
    forward_model, fwd_name, _ = get_best_forward_model(
        ckpt_dir, 
        n_continuous=stats["n_continuous"], 
        n_wavelengths=stats["n_wavelengths"], 
        n_harmonics=stats["n_harmonics"],
        al_iter=args.al_iter,
        force_model_name=args.force_forward_model
    )
    
    if forward_model is None:
        print("Error: No forward model found.")
        return
        
    forward_model.to(device)
    geo_min = stats["geo_min"].to(device)
    geo_max = stats["geo_max"].to(device)
    
    if args.expand_amps is not None:
        n_harmonics = stats["n_harmonics"]
        # In models.py, geometry is packed as [amp0, phase0, amp1, phase1...]
        # Amplitudes are at even indices: 0, 2, 4... up to 2*n_harmonics
        geo_max[0:2*n_harmonics:2] = args.expand_amps
        print(f"[*] Expanded Amplitude Bounds to {args.expand_amps} nm for Active Learning Proposal!")
        
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
        trained_materials = stats["materials"]
        valid_mat_indices = [list(MATERIAL_LIBRARY.keys()).index(m) for m in trained_materials]
        
        # 1. Propose candidates via Surrogate Optimization (Batched across ALL materials)
        forward_model.eval()
        opt = BatchedSurrogateOptimizer(forward_model, geo_min, geo_max, device=device, nx=128, h_val=args.h_val, inc_val=args.inc_val)
        
        proposed_geos_per_mat = {m: [] for m in trained_materials}
        bands_per_proposal = []
        
        for _ in tqdm(range(args.proposals_per_mat), desc=f"Proposing targets (Batched {args.mode})"):
            bands = get_random_bands()
            bands_per_proposal.append(bands)
            if args.mode == "de":
                opt_res = opt.optimize_de(bands, allowed_materials=valid_mat_indices, pop_size=args.restarts, generations=args.steps, F=0.8, CR=0.9, top_k=1, show_progress=True)
            else:
                opt_res = opt.optimize_geometry(bands, allowed_materials=valid_mat_indices, n_restarts=args.restarts, steps=args.steps, lr=0.1, top_k=1, show_progress=True)
            
            for r in opt_res["top_results"]:
                mat_idx = r["material_idx"]
                mat_name = list(MATERIAL_LIBRARY.keys())[mat_idx]
                proposed_geos_per_mat[mat_name].append(r["geometry"])
                
        for mat_name in trained_materials:
            print(f"\n--- Material: {mat_name} ---")
            mat_idx = list(MATERIAL_LIBRARY.keys()).index(mat_name)
            
            proposed_geos = torch.stack(proposed_geos_per_mat[mat_name]).cpu()
            
            # 2. Oracle Verification (Torcwa)
            true_curves = evaluate_oracle(proposed_geos, mat_name, stats, device)
            
            # Compute MAE between what the surrogate predicted vs physical truth
            true_curves_gpu = true_curves.to(device)
            proposed_geos_gpu = proposed_geos.to(device)
            with torch.no_grad():
                surr_preds = forward_model(proposed_geos_gpu, torch.full((len(proposed_geos_gpu),), mat_idx, dtype=torch.long, device=device))
                mae_discrepancy = torch.nn.functional.l1_loss(surr_preds, true_curves_gpu).item()
                
            print(f"Oracle Verification Complete. Surrogate-Physics Discrepancy (MAE): {mae_discrepancy:.4f}")
            
            # --- PLOT & LOG DASHBOARD ---
            n_plots = len(proposed_geos)
            
            log_entries = []
            log_file = al_ckpt_dir / "active_learning_progress.jsonl"
            
            fig, axes = plt.subplots(n_plots, 2, figsize=(10, 3 * n_plots), squeeze=False, layout="constrained")
            wls = WAVELENGTHS
            cmap = plt.cm.viridis
            c_surr, c_physics = cmap(0.5), cmap(0.8)
            for p_idx in range(n_plots):
                target_bands = bands_per_proposal[p_idx]
                target_tensor, _ = opt._get_target_and_mask(target_bands)
                target_np = target_tensor[0].cpu().numpy()
                
                # P-pol
                ax_p = axes[p_idx, 0]
                
                if not target_bands:
                    ax_p.axvspan(wls[0], wls[-1], color='gray', alpha=0.15)
                else:
                    for bmin, bmax in target_bands:
                        ax_p.axvspan(bmin, bmax, color='gray', alpha=0.15)
                        
                ax_p.plot(wls, target_np[:len(wls)], "k--", lw=3, label="Target")
                ax_p.plot(wls, true_curves[p_idx][:len(wls)].numpy(), label="Torcwa", color=c_physics, lw=2.5)
                ax_p.plot(wls, surr_preds[p_idx][:len(wls)].cpu().numpy(), label="Surrogate", color=c_surr, linestyle="-", lw=2)
                
                ax_p.set_title(f"Proposal {p_idx+1} P-pol | MAE: {torch.nn.functional.l1_loss(surr_preds[p_idx], true_curves_gpu[p_idx]).item():.4f}")
                ax_p.grid(True, alpha=0.3)
                ax_p.set_ylabel("Absorptance")
                ax_p.set_ylim(-0.05, 1.05)
                
                # S-pol
                ax_s = axes[p_idx, 1]
                
                if not target_bands:
                    ax_s.axvspan(wls[0], wls[-1], color='gray', alpha=0.15)
                else:
                    for bmin, bmax in target_bands:
                        ax_s.axvspan(bmin, bmax, color='gray', alpha=0.15)
                        
                ax_s.plot(wls, target_np[len(wls):], "k--", lw=3, label="Target")
                ax_s.plot(wls, true_curves[p_idx][len(wls):].numpy(), label="Torcwa", color=c_physics, lw=2.5)
                ax_s.plot(wls, surr_preds[p_idx][len(wls):].cpu().numpy(), label="Surrogate", color=c_surr, linestyle="-", lw=2)
                
                ax_s.set_title(f"Proposal {p_idx+1} S-pol")
                ax_s.grid(True, alpha=0.3)
                ax_s.set_ylim(-0.05, 1.05)
                
                if p_idx == n_plots - 1:
                    ax_p.set_xlabel("Wavelength (nm)")
                    ax_s.set_xlabel("Wavelength (nm)")
                if p_idx == 0:
                    ax_p.legend()
                    ax_s.legend()
                    
                # Compute and log metrics
                mask = np.zeros(len(WAVELENGTHS), dtype=bool)
                if not target_bands:
                    mask = np.ones(len(WAVELENGTHS), dtype=bool)
                else:
                    for bmin, bmax in target_bands:
                        mask |= (WAVELENGTHS >= bmin) & (WAVELENGTHS <= bmax)
                
                # Combine P and S mask
                full_mask = np.concatenate([mask, mask])
                true_curve_np = true_curves[p_idx].numpy()
                surr_curve_np = surr_preds[p_idx].cpu().numpy()
                
                oracle_abs = float(np.mean(true_curve_np[full_mask]))
                surrogate_abs = float(np.mean(surr_curve_np[full_mask]))
                
                log_entries.append(json.dumps({
                    "iteration": al_iterations,
                    "material": mat_name,
                    "proposal_idx": p_idx,
                    "target_bands": target_bands,
                    "surrogate_abs": surrogate_abs,
                    "oracle_abs": oracle_abs,
                    "mae": float(np.mean(np.abs(true_curve_np - surr_curve_np))),
                    "geometry": proposed_geos[p_idx].numpy().tolist()
                }))
                    
            import re
            base_stem = re.sub(r"_al\d+", "", Path(fwd_name).stem)
            fig.suptitle(f"Active Learning Discrepancy (Iter {al_iterations}) | {mat_name}", fontsize=16)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / f"al_discrepancy_{mat_name}_iter{al_iterations}.png")
            plt.close(fig)
            
            with open(log_file, "a") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
            # ----------------------
            
            # 3. Save new data
            save_al_dataset(proposed_geos, true_curves, mat_name, al_dir)
            
        # 4. Finetune Surrogate (across all materials simultaneously)
        print("\n--- Finetuning Surrogate (All Materials) ---")
        forward_model = finetune_surrogate(forward_model, trained_materials, stats, al_dir)
        
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
