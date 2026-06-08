
def process_sample(i, h_arr, inc_ang_arr, amps_arr, phases_arr, wavelengths, args):
    import torch
    import numpy as np
    from Utils.utils import get_absorptance_curve, RCWAConfig, geo_dtype
    torch.set_num_threads(1)
    device = torch.device('cpu')
    
    h = h_arr[i]
    inc_ang_deg = inc_ang_arr[i]
    
    params_x_data = [[amps_arr[i, j], phases_arr[i, j]] for j in range(5)]
    params_x = torch.tensor(params_x_data, dtype=geo_dtype, device=device)
    
    base_config = RCWAConfig(
        grating_period=args.grating_period, h=float(h), order_N=args.order_N,
        n_layers=args.num_layers, height_per_layer=args.height_per_layer,
        nx=args.nx, ny=1,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    base_config.inc_ang = 1e-3 * np.pi/180
    base_config.azi_ang = 1e-3 * np.pi/180
    A_film_norm, A_grat_norm = get_absorptance_curve(
        params_x=params_x, params_y=None,
        wavelengths=wavelengths, config=base_config
    )
    
    base_config.inc_ang = (inc_ang_deg + 1e-3) * np.pi/180
    base_config.azi_ang = 1e-3 * np.pi/180
    A_film_obl, A_grat_obl = get_absorptance_curve(
        params_x=params_x, params_y=None,
        wavelengths=wavelengths, config=base_config
    )
    
    return {
        'h': float(h),
        'inc_ang': float(inc_ang_deg),
        'params_x': params_x.cpu(),
        'A_film_normal': A_film_norm.cpu(),
        'A_grating_normal': A_grat_norm.cpu(),
        'A_film_oblique': A_film_obl.cpu(),
        'A_grating_oblique': A_grat_obl.cpu(),
    }

import argparse
import multiprocessing as mp
from functools import partial
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import asdict

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.utils import geo_dtype, RCWAConfig

# Import shared sample generation from generate_dataset.py
from generate_dataset import get_or_create_samples

def main():
    parser = argparse.ArgumentParser(description="Generate LHS Dataset for Inverse Design (CPU parallel)")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Total number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Number of samples to save per .pt file")
    parser.add_argument('--order_N', type=int, default=5,
                        help="Diffraction order")
    parser.add_argument('--n_jobs', type=int, default=4,
                        help="Number of CPU cores")
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
    
    # Load or generate LHS samples (shared with GPU script)
    h_arr, inc_ang_arr, amps_arr, phases_arr = get_or_create_samples(out_dir, args.num_samples, seed=args.seed)
    
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    
    # Save a clean config for metadata
    clean_config = RCWAConfig(
        grating_period=args.grating_period, h=0.0, order_N=args.order_N,
        n_layers=args.num_layers, height_per_layer=args.height_per_layer,
        nx=args.nx, ny=1,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    print(f"Starting LHS dataset generation (CPU parallel, {args.n_jobs} workers): {args.num_samples} samples in {num_batches} batches.")
    print(f"Config: {asdict(clean_config)}")
    sys.stdout.flush()
    
    torch.set_num_threads(1)
    
    for batch_idx in range(num_batches):
        batch_file = os.path.join(out_dir, f"batch_{batch_idx:04d}.pt")
        if os.path.exists(batch_file):
            print(f"Batch {batch_idx:04d} already exists. Skipping...")
            sys.stdout.flush()
            continue
            
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        batch_len = end_idx - start_idx
        
        print(f"\nComputing Batch {batch_idx:04d} (Samples {start_idx} to {end_idx-1})...")
        sys.stdout.flush()
        
        worker = partial(
            process_sample,
            h_arr=h_arr, inc_ang_arr=inc_ang_arr,
            amps_arr=amps_arr, phases_arr=phases_arr,
            wavelengths=wavelengths, args=args
        )
        
        with mp.Pool(processes=args.n_jobs) as pool:
            sample_dicts = list(tqdm(
                pool.imap(worker, range(start_idx, end_idx)),
                total=batch_len,
                desc=f"Batch {batch_idx:04d}",
                mininterval=2.0,
                file=sys.stdout
            ))
        
        # Stack into tensors (same format as generate_dataset.py)
        all_h = torch.tensor([s['h'] for s in sample_dicts])
        all_inc_ang = torch.tensor([s['inc_ang'] for s in sample_dicts])
        all_params_x = torch.stack([s['params_x'] for s in sample_dicts])
        all_A_film_normal = torch.stack([s['A_film_normal'] for s in sample_dicts])
        all_A_grating_normal = torch.stack([s['A_grating_normal'] for s in sample_dicts])
        all_A_film_oblique = torch.stack([s['A_film_oblique'] for s in sample_dicts])
        all_A_grating_oblique = torch.stack([s['A_grating_oblique'] for s in sample_dicts])
        
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
