import argparse
import os
import sys
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataclasses import asdict
from Utils.utils import get_absorptance_curve, geo_dtype, device, RCWAConfig

def get_lhs_samples(num_samples):
    # 13 dimensions: h, inc_ang, azi_ang, 5x amplitudes, 5x phases
    sampler = LatinHypercube(d=13)
    sample = sampler.random(n=num_samples)
    
    # Map from [0, 1] to physical bounds
    h = 500 + 4500 * sample[:, 0]        # 500 nm to 5000 nm
    inc_ang = 0 + 30 * sample[:, 1]      # 0 to 30 degrees
    azi_ang = 0 + 0 * sample[:, 2]      # 0 to 0 degrees for 2D only.
    
    amps = 0 + 10 * sample[:, 3:8]       # 0 to 10 nm max
    phases = 0 + 2 * np.pi * sample[:, 8:13] # 0 to 2*pi
    
    return h, inc_ang, azi_ang, amps, phases

def main():
    parser = argparse.ArgumentParser(description="Generate LHS Dataset for Inverse Design")
    parser.add_argument('--num_samples', type=int, default=5000, help="Total number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=100, help="Number of samples to save per .pt file")
    parser.add_argument('--order_N', type=int, default=10, help="Diffraction order")
    parser.add_argument('--num_layers', type=int, default=10, help="Number of staircase layers")
    parser.add_argument('--height_per_layer', type=float, default=None, help="Overrides num_layers to fix grating resolution")
    parser.add_argument('--grating_period', type=float, default=1000.0, help="Grating period (nm)")
    parser.add_argument('--nx', type=int, default=5000, help="Grid resolution")
    parser.add_argument('--grating_material', type=str, default='Si', help="Material for the grating layer (e.g. Si, TiO2, Si3N4)")
    parser.add_argument('--no_reflector', action='store_true', help="Disable the bottom reflector")
    parser.add_argument('--reflector_type', type=str, default='pec', help="Reflector type (e.g., pec, Ag)")
    parser.add_argument('--no_subpixel', action='store_true', help="Disable subpixel smoothing")
    
    args = parser.parse_args()
    
    out_dir = os.path.join(project_root, 'Data', 'LHS_Dataset')
    os.makedirs(out_dir, exist_ok=True)
    
    # Wavelengths: 300 to 1100 nm in 5 nm steps (161 steps)
    wavelengths = torch.linspace(300, 1100, 161, dtype=torch.float64) + 1e-3
    
    # Generate LHS
    h_arr, inc_ang_arr, azi_ang_arr, amps_arr, phases_arr = get_lhs_samples(args.num_samples)
    
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    
    print(f"Starting LHS dataset generation: {args.num_samples} samples in {num_batches} batches.")
    
    for batch_idx in range(num_batches):
        batch_file = os.path.join(out_dir, f"batch_{batch_idx:04d}.pt")
        if os.path.exists(batch_file):
            print(f"Batch {batch_idx:04d} already exists. Skipping...")
            continue
            
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        
        batch_data = []
        
        print(f"\nComputing Batch {batch_idx:04d} (Samples {start_idx} to {end_idx-1})...")
        for i in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx:04d}"):
            h = h_arr[i]
            inc_ang_deg = inc_ang_arr[i]
            azi_ang_deg = azi_ang_arr[i]
            
            # Format params_x as (5, 2) tensor
            params_x_data = [[amps_arr[i, j], phases_arr[i, j]] for j in range(5)]
            params_x = torch.tensor(params_x_data, dtype=geo_dtype, device=device)
            
            # Base Config
            base_config = RCWAConfig(
                grating_period=args.grating_period, h=float(h), order_N=args.order_N, 
                n_layers=args.num_layers, height_per_layer=args.height_per_layer,
                nx=args.nx, ny=1, add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
                subpixel=not args.no_subpixel, grating_material=args.grating_material
            )
            
            # 1. Normal Incidence Calculation
            base_config.inc_ang = 1e-3 * np.pi/180
            base_config.azi_ang = 1e-3 * np.pi/180
            A_film_norm, A_grat_norm = get_absorptance_curve(params_x=params_x, params_y=None, wavelengths=wavelengths, config=base_config)
            
            # 2. Oblique Incidence Calculation
            base_config.inc_ang = (inc_ang_deg + 1e-3) * np.pi/180
            base_config.azi_ang = (azi_ang_deg + 1e-3) * np.pi/180
            A_film_obl, A_grat_obl = get_absorptance_curve(params_x=params_x, params_y=None, wavelengths=wavelengths, config=base_config)
            
            sample_dict = {
                'sample_id': i,
                'h': float(h),
                'inc_ang': float(inc_ang_deg),
                'azi_ang': float(azi_ang_deg),
                'params_x': params_x.cpu(),
                'A_film_normal': A_film_norm.cpu(),
                'A_grating_normal': A_grat_norm.cpu(),
                'A_film_oblique': A_film_obl.cpu(),
                'A_grating_oblique': A_grat_obl.cpu(),
            }
            batch_data.append(sample_dict)
            
        # Save batch file dynamically
        save_dict = {
            'metadata': {
                'wavelengths': wavelengths.cpu(),
                'config': asdict(base_config)
            },
            'samples': batch_data
        }
        torch.save(save_dict, batch_file)
        print(f"Saved {batch_file}")

if __name__ == "__main__":
    main()
