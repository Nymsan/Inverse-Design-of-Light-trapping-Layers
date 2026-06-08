import argparse
import os
import sys
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
from dataclasses import asdict

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.utils import get_absorptance_curve, geo_dtype, RCWAConfig
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_lhs_samples(num_samples, seed=42):
    """Generate and return LHS samples as numpy arrays."""
    # 12 dimensions: h, inc_ang, 5x amplitudes, 5x phases
    sampler = LatinHypercube(d=12, seed=seed)
    sample = sampler.random(n=num_samples)
    
    # Map from [0, 1] to physical bounds
    h = 500 + 4500 * sample[:, 0]        # 500 nm to 5000 nm
    inc_ang = 0 + 30 * sample[:, 1]      # 0 to 30 degrees
    
    amps = 0 + 10 * sample[:, 2:7]       # 0 to 10 nm max
    phases = 0 + 2 * np.pi * sample[:, 7:12] # 0 to 2*pi
    
    return h, inc_ang, amps, phases

def get_or_create_samples(out_dir, num_samples, seed=42):
    """Load existing samples or generate and save new ones.
    
    This ensures both GPU and CPU-parallel scripts use identical samples
    when writing to the same output directory.
    """
    samples_file = os.path.join(out_dir, '_lhs_samples.npz')
    
    if os.path.exists(samples_file):
        data = np.load(samples_file)
        if len(data['h']) == num_samples:
            print(f"Loaded existing LHS samples from {samples_file}")
            sys.stdout.flush()
            return data['h'], data['inc_ang'], data['amps'], data['phases']
        else:
            print(f"WARNING: Existing samples have {len(data['h'])} entries but {num_samples} requested. Regenerating...")
            sys.stdout.flush()
    
    h, inc_ang, amps, phases = generate_lhs_samples(num_samples, seed=seed)
    np.savez(samples_file, h=h, inc_ang=inc_ang, amps=amps, phases=phases)
    print(f"Generated and saved LHS samples to {samples_file}")
    sys.stdout.flush()
    return h, inc_ang, amps, phases

def main():
    parser = argparse.ArgumentParser(description="Generate LHS Dataset for Inverse Design")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Total number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Number of samples to save per .pt file")
    parser.add_argument('--order_N', type=int, default=10,
                        help="Diffraction order")
    parser.add_argument('--num_layers', type=int, default=10,
                        help="Number of staircase layers")
    parser.add_argument('--height_per_layer', type=float, default=None,
                        help="Overrides num_layers to fix grating resolution")
    parser.add_argument('--grating_period', type=float, default=1000.0,
                        help="Grating period (nm)")
    parser.add_argument('--nx', type=int, default=5000,
                        help="Grid resolution")
    parser.add_argument('--grating_material', type=str, default='Si',
                        help="Material for the grating layer (e.g. Si, TiO2, Si3N4)")
    parser.add_argument('--no_reflector', action='store_true',
                        help="Disable the bottom reflector")
    parser.add_argument('--reflector_type', type=str, default='pec',
                        help="Reflector type (e.g., pec, Ag)")
    parser.add_argument('--no_subpixel', action='store_true',
                        help="Disable subpixel smoothing")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for LHS sampling")
    
    args = parser.parse_args()
    
    out_dir = os.path.join(project_root, 'Data', f'LHS_Dataset_{args.grating_material}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Wavelengths: 300 to 1100 nm in 5 nm steps (161 steps)
    wavelengths = torch.linspace(300, 1100, 161, dtype=torch.float64) + 1e-3
    
    # Load or generate LHS samples (shared with cpu_parallel script)
    h_arr, inc_ang_arr, amps_arr, phases_arr = get_or_create_samples(out_dir, args.num_samples, seed=args.seed)
    
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    
    # Save a clean config for metadata (before any angle mutation)
    clean_config = RCWAConfig(
        grating_period=args.grating_period, h=0.0, order_N=args.order_N,
        n_layers=args.num_layers, height_per_layer=args.height_per_layer,
        nx=args.nx, ny=1,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    print(f"Starting LHS dataset generation: {args.num_samples} samples in {num_batches} batches.")
    print(f"Config: {asdict(clean_config)}")
    sys.stdout.flush()
    
    for batch_idx in range(num_batches):
        batch_file = os.path.join(out_dir, f"batch_{batch_idx:04d}.pt")
        if os.path.exists(batch_file):
            print(f"Batch {batch_idx:04d} already exists. Skipping...")
            sys.stdout.flush()
            continue
            
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        batch_len = end_idx - start_idx
        
        # Pre-allocate stacked tensors for DataLoader-friendly format
        all_h = torch.zeros(batch_len)
        all_inc_ang = torch.zeros(batch_len)
        all_params_x = torch.zeros(batch_len, 5, 2)
        all_A_film_normal = torch.zeros(batch_len, len(wavelengths), 2)
        all_A_grating_normal = torch.zeros(batch_len, len(wavelengths), 2)
        all_A_film_oblique = torch.zeros(batch_len, len(wavelengths), 2)
        all_A_grating_oblique = torch.zeros(batch_len, len(wavelengths), 2)
        
        print(f"\nComputing Batch {batch_idx:04d} (Samples {start_idx} to {end_idx-1})...")
        sys.stdout.flush()
        for local_i, i in enumerate(tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx:04d}", mininterval=2.0, file=sys.stdout)):
            h = h_arr[i]
            inc_ang_deg = inc_ang_arr[i]
            
            # Format params_x as (5, 2) tensor
            params_x_data = [[amps_arr[i, j], phases_arr[i, j]] for j in range(5)]
            params_x = torch.tensor(params_x_data, dtype=geo_dtype, device=default_device)
            
            # Base Config
            base_config = RCWAConfig(
                grating_period=args.grating_period, h=float(h), order_N=args.order_N,
                n_layers=args.num_layers, height_per_layer=args.height_per_layer,
                nx=args.nx, ny=1,
                add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
                subpixel=not args.no_subpixel, grating_material=args.grating_material
            )
            
            # 1. Normal Incidence Calculation
            base_config.inc_ang = 1e-3 * np.pi/180
            base_config.azi_ang = 1e-3 * np.pi/180
            A_film_norm, A_grat_norm = get_absorptance_curve(
                params_x=params_x, params_y=None,
                wavelengths=wavelengths, config=base_config
            )
            
            # 2. Oblique Incidence Calculation
            base_config.inc_ang = (inc_ang_deg + 1e-3) * np.pi/180
            base_config.azi_ang = 1e-3 * np.pi/180
            A_film_obl, A_grat_obl = get_absorptance_curve(
                params_x=params_x, params_y=None,
                wavelengths=wavelengths, config=base_config
            )
            
            # Store into pre-allocated tensors
            all_h[local_i] = float(h)
            all_inc_ang[local_i] = float(inc_ang_deg)
            all_params_x[local_i] = params_x.cpu()
            all_A_film_normal[local_i] = A_film_norm.cpu()
            all_A_grating_normal[local_i] = A_grat_norm.cpu()
            all_A_film_oblique[local_i] = A_film_obl.cpu()
            all_A_grating_oblique[local_i] = A_grat_obl.cpu()
            
        # Save batch as stacked tensors (DataLoader-friendly)
        save_dict = {
            'metadata': {
                'wavelengths': wavelengths.cpu(),
                'config': asdict(clean_config),
                'batch_idx': batch_idx,
                'sample_ids': list(range(start_idx, end_idx)),
            },
            'h': all_h,
            'inc_ang': all_inc_ang,
            'params_x': all_params_x,
            'A_film_normal': all_A_film_normal,
            'A_grating_normal': all_A_grating_normal,
            'A_film_oblique': all_A_film_oblique,
            'A_grating_oblique': all_A_grating_oblique,
        }
        torch.save(save_dict, batch_file)
        print(f"Saved {batch_file}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
