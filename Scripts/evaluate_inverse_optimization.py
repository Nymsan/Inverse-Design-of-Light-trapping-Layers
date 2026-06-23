import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.models import MATERIAL_LIBRARY
from Utils.utils import get_absorptance_curve, RCWAConfig, build_profile, recover_geometry_from_profile

class JointOptimizer:
    def __init__(self, forward_model, inverse_model, model_type, geo_min, geo_max, device="cuda"):
        self.forward_model = forward_model.to(device)
        self.inverse_model = inverse_model.to(device)
        self.forward_model.eval()
        self.inverse_model.eval()
        for p in self.forward_model.parameters(): p.requires_grad = False
        for p in self.inverse_model.parameters(): p.requires_grad = False
        
        self.model_type = model_type
        self.device = device
        self.geo_min = geo_min.to(device)
        self.geo_max = geo_max.to(device)

    def optimize(self, material_idx, n_wavelengths=322, steps=1000, lr=0.01, lambda_reg=5.0, 
                 bands=None, h_val=None, inc_val=None, robust_opt=False):
        # We optimize a single curve target, and optionally a single Z vector
        
        # Start Y_target as all 1s (ideal case) inside bands, or broadband
        Y_target_np = np.ones((1, n_wavelengths))
        if bands is not None:
            wavelengths = np.linspace(300, 1100, n_wavelengths // 2)
            band_mask = np.zeros(len(wavelengths), dtype=bool)
            for bmin, bmax in bands:
                band_mask |= (wavelengths >= bmin) & (wavelengths <= bmax)
            full_mask = np.concatenate([band_mask, band_mask])
            Y_target_np[:, ~full_mask] = 0.0
            
        Y_target = torch.tensor(Y_target_np, dtype=torch.float32, device=self.device)
        Y_target = nn.Parameter(Y_target)
        
        params_to_opt = [Y_target]
        
        if not robust_opt and self.model_type in ["gen_tandem", "cvae"]:
            # Z dimension is 64 for gen_tandem / cvae by default, but let's grab it from the model
            if self.model_type == "gen_tandem":
                z_dim = self.inverse_model.latent_dim
            else: # cvae
                z_dim = self.inverse_model.latent_dim
            Z = torch.randn(1, z_dim, device=self.device)
            Z = nn.Parameter(Z)
            params_to_opt.append(Z)
        else:
            Z = None
            
        mat_id = torch.tensor([material_idx], dtype=torch.long, device=self.device)
        
        optimizer = optim.Adam(params_to_opt, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        pbar = tqdm(range(steps), desc=f"Optimizing {self.model_type} (mat {material_idx})")
        best_geo = None
        best_abs = -float('inf')
        
        for step in pbar:
            optimizer.zero_grad()
            
            # 1. Map target curve to Geometry
            if self.model_type == "tandem":
                geo, _, _ = self.inverse_model(Y_target, tau=0.1, hard=True)
            elif self.model_type == "gen_tandem":
                if robust_opt:
                    Z_stochastic = torch.randn(1, self.inverse_model.latent_dim, device=self.device)
                    geo, _, _ = self.inverse_model.decoder(Y_target, Z_stochastic, tau=0.1, hard=True)
                else:
                    geo, _, _ = self.inverse_model.decoder(Y_target, Z, tau=0.1, hard=True)
            elif self.model_type == "cvae":
                Z_y = self.inverse_model.spectrum_encoder(Y_target)
                if robust_opt:
                    # Sample Z around Z_y like normal inference
                    direction = torch.nn.functional.normalize(torch.randn(1, Z_y.shape[-1], device=self.device), p=2, dim=-1)
                    radii = torch.rand(1, 1, device=self.device) * self.inverse_model.margin_radius
                    Z_stochastic = Z_y + radii * direction
                    geo, _, _ = self.inverse_model.geometry_decoder(Z_stochastic, tau=0.1, hard=True)
                else:
                    geo, _, _ = self.inverse_model.geometry_decoder(Z, tau=0.1, hard=True)
            
            # Clamp geometry to physical bounds to prevent forward model from crashing
            geo_clamped = torch.clamp(geo, self.geo_min, self.geo_max)
            
            # 2. Map Geometry to Predicted Curve
            Y_pred = self.forward_model(geo_clamped, mat_id)
            
            # 3. Compute Loss
            if bands is not None:
                # Maximize only in-band absorptance
                abs_loss = -Y_pred[:, full_mask].mean()
            else:
                abs_loss = -Y_pred.mean()
            
            # Regularization: Y_target must match Y_pred (so Inverse model operates in distribution)
            if self.model_type == "cvae" and not robust_opt:
                # For Joint CVAE, we also want Z to map to Y_target's embedding Z_y
                realism_penalty = torch.nn.functional.mse_loss(Z_y, Z) + torch.nn.functional.mse_loss(Y_target, Y_pred)
            else:
                realism_penalty = torch.nn.functional.mse_loss(Y_target, Y_pred)
                
            # Physics Constraints (Force Inverse model to output desired h and inc_ang)
            physics_penalty = 0.0
            if h_val is not None:
                # normalize penalty based on typical height range
                physics_penalty += torch.nn.functional.mse_loss(geo[:, -2], torch.tensor([h_val], device=self.device)) / (1000.0**2)
            if inc_val is not None:
                physics_penalty += torch.nn.functional.mse_loss(geo[:, -1], torch.tensor([inc_val], device=self.device))
                
            loss = abs_loss + lambda_reg * realism_penalty + 50.0 * physics_penalty
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                Y_target.clamp_(0.0, 1.0) # Curves are between 0 and 1
                curr_abs = -abs_loss.item() # Use the in-band or broadband abs computed above
                if curr_abs > best_abs:
                    best_abs = curr_abs
                    best_geo = geo_clamped.detach().clone()
                    
                pbar.set_postfix({'Abs': f"{curr_abs:.4f}", 'Reg': f"{realism_penalty.item():.4f}"})
                
        return best_geo, best_abs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory with trained models")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--bands", nargs="+", type=float, help="Pairs of wavelength bands to optimize, e.g., --bands 500 750 800 900")
    parser.add_argument("--h_val", type=float, default=None, help="Constrain grating height (nm)")
    parser.add_argument("--inc_val", type=float, default=None, help="Constrain incidence angle (deg)")
    parser.add_argument("--robust", action="store_true", help="Use robust optimization (resample Z every step) instead of joint optimization")
    args = parser.parse_args()
    
    # Parse bands
    bands = None
    if args.bands:
        if len(args.bands) % 2 != 0:
            raise ValueError("Bands must be pairs of min max wavelengths.")
        bands = [(args.bands[i], args.bands[i+1]) for i in range(0, len(args.bands), 2)]
        
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats = torch.load(ckpt_dir / "dataset_stats.pt", map_location="cpu", weights_only=False)
    materials = stats["materials"]
    
    # Load Forward Model
    fwd_path = ckpt_dir / "skip_cnn.pt"
    if not fwd_path.exists():
        print("Forward model not found!")
        return
    fwd_model = torch.load(fwd_path, map_location=device, weights_only=False)
    
    models_to_test = ["tandem.pt", "gen_tandem.pt", "cvae.pt"]
    
    out_dir = ckpt_dir / "evaluation" / "inverse_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for m_file in models_to_test:
        m_path = ckpt_dir / m_file
        if not m_path.exists():
            continue
            
        print(f"\nLoading {m_file}...")
        inv_model = torch.load(m_path, map_location=device, weights_only=False)
        model_type = m_file.split('.')[0]
        
        opt = JointOptimizer(fwd_model, inv_model, model_type, stats["geo_min"], stats["geo_max"], device)
        
        for mat_idx, mat_name in enumerate(materials):
            best_geo, best_abs = opt.optimize(mat_idx, n_wavelengths=stats["n_wavelengths"], steps=args.steps, 
                                              bands=bands, h_val=args.h_val, inc_val=args.inc_val, robust_opt=args.robust)
            
            mode_str = "Robust" if args.robust else "Joint"
            print(f"[{model_type} ({mode_str}) - {mat_name}] Optimized Forward Predicted Abs: {best_abs:.4f}")
            
            # Here we could run Torcwa to verify the geometry physically, just like naive optimization!
            geo_np = best_geo[0].cpu().numpy()
            
            res_file = out_dir / f"{model_type}_{mat_name}_geo.npy"
            np.save(res_file, geo_np)
            
    print(f"\nAll optimizations complete! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
