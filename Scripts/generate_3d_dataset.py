def process_sample_3d(i, h_arr, inc_ang_arr, azi_ang_arr, wl_arr, amps_x_arr, phases_x_arr, amps_y_arr, phases_y_arr, args):
    import torch
    import numpy as np
    from Utils.utils import get_absorptance_curve, RCWAConfig, geo_dtype
    torch.set_num_threads(1)
    device = torch.device('cpu')
    
    h = h_arr[i]
    inc_ang_deg = inc_ang_arr[i]
    azi_ang_deg = azi_ang_arr[i]
    wl = wl_arr[i]
    
    params_x_data = [[amps_x_arr[i, j], phases_x_arr[i, j]] for j in range(5)]
    params_y_data = [[amps_y_arr[i, j], phases_y_arr[i, j]] for j in range(5)]
    params_x = torch.tensor(params_x_data, dtype=geo_dtype, device=device)
    params_y = torch.tensor(params_y_data, dtype=geo_dtype, device=device)
    
    wavelength_tensor = torch.tensor([wl], dtype=torch.float64) + 1e-3
    
    base_config = RCWAConfig(
        grating_period=args.grating_period, grating_period_y=args.grating_period_y,
        h=float(h), order_N=args.order_N, order_N_y=args.order_N_y,
        n_layers=args.num_layers, height_per_layer=args.height_per_layer,
        nx=args.nx, ny=args.ny,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    base_config.inc_ang = 1e-3 * np.pi/180
    base_config.azi_ang = 1e-3 * np.pi/180
    A_film_norm, A_grat_norm = get_absorptance_curve(
        params_x=params_x, params_y=params_y,
        wavelengths=wavelength_tensor, config=base_config
    )
    
    base_config.inc_ang = (inc_ang_deg + 1e-3) * np.pi/180
    base_config.azi_ang = (azi_ang_deg + 1e-3) * np.pi/180
    A_film_obl, A_grat_obl = get_absorptance_curve(
        params_x=params_x, params_y=params_y,
        wavelengths=wavelength_tensor, config=base_config
    )
    
    return {
        'wavelength': float(wl),
        'h': float(h),
        'inc_ang': float(inc_ang_deg),
        'azi_ang': float(azi_ang_deg),
        'params_x': params_x.cpu(),
        'params_y': params_y.cpu(),
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
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
from dataclasses import asdict

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.utils import geo_dtype, RCWAConfig
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lhs_samples(num_samples, seed=42):
    sampler = LatinHypercube(d=24, seed=seed)
    sample = sampler.random(n=num_samples)
    
    h = 500 + 4500 * sample[:, 0]            # 500 nm to 5000 nm
    inc_ang = 0 + 30 * sample[:, 1]          # 0 to 30 degrees
    azi_ang = 0 + 360 * sample[:, 2]         # 0 to 360 degrees
    wavelengths = 300 + 800 * sample[:, 3]   # 300 nm to 1100 nm
    
    amps_x = 0 + 10 * sample[:, 4:9]         # 0 to 10 nm max
    phases_x = 0 + 2 * np.pi * sample[:, 9:14] # 0 to 2*pi
    
    amps_y = 0 + 10 * sample[:, 14:19]       # 0 to 10 nm max
    phases_y = 0 + 2 * np.pi * sample[:, 19:24] # 0 to 2*pi
    
    return h, inc_ang, azi_ang, wavelengths, amps_x, phases_x, amps_y, phases_y

def get_or_create_samples_3d(out_dir, num_samples, seed=42):
    samples_file = os.path.join(out_dir, '_lhs_samples_3d.npz')
    
    if os.path.exists(samples_file):
        data = np.load(samples_file)
        if len(data['h']) == num_samples:
            print(f"Loaded existing LHS samples from {samples_file}")
            sys.stdout.flush()
            return data['h'], data['inc_ang'], data['azi_ang'], data['wavelengths'], data['amps_x'], data['phases_x'], data['amps_y'], data['phases_y']
        else:
            print(f"WARNING: Existing samples have {len(data['h'])} entries but {num_samples} requested. Regenerating...")
            sys.stdout.flush()
            
    h, inc_ang, azi_ang, wavelengths, amps_x, phases_x, amps_y, phases_y = get_lhs_samples(num_samples, seed=seed)
    np.savez(samples_file, h=h, inc_ang=inc_ang, azi_ang=azi_ang, wavelengths=wavelengths, amps_x=amps_x, phases_x=phases_x, amps_y=amps_y, phases_y=phases_y)
    print(f"Generated and saved LHS samples to {samples_file}")
    sys.stdout.flush()
    return h, inc_ang, azi_ang, wavelengths, amps_x, phases_x, amps_y, phases_y

def main():
    parser = argparse.ArgumentParser(description="Generate 3D LHS Dataset for Inverse Design (CPU parallel)")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Total number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Number of samples to save per .pt file")
    parser.add_argument('--order_N', type=int, default=10,
                        help="Diffraction order X")
    parser.add_argument('--order_N_y', type=int, default=10,
                        help="Diffraction order Y")
    parser.add_argument('--n_jobs', type=int, default=4,
                        help="Number of CPU cores")
    parser.add_argument('--num_layers', type=int, default=10,
                        help="Number of staircase layers")
    parser.add_argument('--height_per_layer', type=float, default=None,
                        help="Overrides num_layers to fix grating resolution")
    parser.add_argument('--grating_period', type=float, default=1000.0,
                        help="Grating period X (nm)")
    parser.add_argument('--grating_period_y', type=float, default=1000.0,
                        help="Grating period Y (nm)")
    parser.add_argument('--nx', type=int, default=500,
                        help="Grid resolution X")
    parser.add_argument('--ny', type=int, default=500,
                        help="Grid resolution Y")
    parser.add_argument('--grating_material', type=str, default='Si',
                        help="Material for the grating layer (e.g. Si, TiO2, Si3N4)")
    parser.add_argument('--no_reflector', action='store_true',
                        help="Disable the bottom reflector")
    parser.add_argument('--reflector_type', type=str, default='pec',
                        help="Reflector type (e.g., pec, Ag)")
    parser.add_argument('--no_subpixel', action='store_true',
                        help="Disable subpixel smoothing")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for LHS sampling")
    
    args = parser.parse_args()
    
    out_dir = os.path.join(project_root, 'Data', f'LHS_3D_Dataset_{args.grating_material}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate LHS
    h_arr, inc_ang_arr, azi_ang_arr, wl_arr, amps_x_arr, phases_x_arr, amps_y_arr, phases_y_arr = get_or_create_samples_3d(out_dir, args.num_samples, seed=args.seed)
    
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    
    # Save a clean config for metadata (before any angle mutation)
    clean_config = RCWAConfig(
        grating_period=args.grating_period, grating_period_y=args.grating_period_y,
        h=0.0, order_N=args.order_N, order_N_y=args.order_N_y,
        n_layers=args.num_layers, height_per_layer=args.height_per_layer,
        nx=args.nx, ny=args.ny,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    print(f"Starting 3D LHS dataset generation (CPU parallel, {args.n_jobs} workers): {args.num_samples} samples in {num_batches} batches.")
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
            process_sample_3d,
            h_arr=h_arr, inc_ang_arr=inc_ang_arr, azi_ang_arr=azi_ang_arr,
            wl_arr=wl_arr, amps_x_arr=amps_x_arr, phases_x_arr=phases_x_arr,
            amps_y_arr=amps_y_arr, phases_y_arr=phases_y_arr, args=args
        )
        
        with mp.Pool(processes=args.n_jobs) as pool:
            sample_dicts = list(tqdm(
                pool.imap(worker, range(start_idx, end_idx)),
                total=batch_len,
                desc=f"Batch {batch_idx:04d}",
                mininterval=2.0,
                file=sys.stdout
            ))
            
        # Stack into tensors
        all_wavelength = torch.tensor([s['wavelength'] for s in sample_dicts])
        all_h = torch.tensor([s['h'] for s in sample_dicts])
        all_inc_ang = torch.tensor([s['inc_ang'] for s in sample_dicts])
        all_azi_ang = torch.tensor([s['azi_ang'] for s in sample_dicts])
        all_params_x = torch.stack([s['params_x'] for s in sample_dicts])
        all_params_y = torch.stack([s['params_y'] for s in sample_dicts])
        all_A_film_normal = torch.stack([s['A_film_normal'] for s in sample_dicts])
        all_A_grating_normal = torch.stack([s['A_grating_normal'] for s in sample_dicts])
        all_A_film_oblique = torch.stack([s['A_film_oblique'] for s in sample_dicts])
        all_A_grating_oblique = torch.stack([s['A_grating_oblique'] for s in sample_dicts])
        
        # Save batch as stacked tensors (DataLoader-friendly)
        save_dict = {
            'metadata': {
                'config': asdict(clean_config),
                'batch_idx': batch_idx,
                'sample_ids': list(range(start_idx, end_idx)),
            },
            'wavelength': all_wavelength,
            'h': all_h,
            'inc_ang': all_inc_ang,
            'azi_ang': all_azi_ang,
            'params_x': all_params_x,
            'params_y': all_params_y,
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
