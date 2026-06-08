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

def get_lhs_samples(num_samples):
    # 24 dimensions: h, inc_ang, azi_ang, wavelength, 5x amps_x, 5x phases_x, 5x amps_y, 5x phases_y
    sampler = LatinHypercube(d=24)
    sample = sampler.random(n=num_samples)
    
    # Map from [0, 1] to physical bounds
    h = 500 + 4500 * sample[:, 0]            # 500 nm to 5000 nm
    inc_ang = 0 + 30 * sample[:, 1]          # 0 to 30 degrees
    azi_ang = 0 + 360 * sample[:, 2]         # 0 to 360 degrees
    wavelengths = 300 + 800 * sample[:, 3]   # 300 nm to 1100 nm
    
    amps_x = 0 + 10 * sample[:, 4:9]         # 0 to 10 nm max
    phases_x = 0 + 2 * np.pi * sample[:, 9:14] # 0 to 2*pi
    
    amps_y = 0 + 10 * sample[:, 14:19]       # 0 to 10 nm max
    phases_y = 0 + 2 * np.pi * sample[:, 19:24] # 0 to 2*pi
    
    return h, inc_ang, azi_ang, wavelengths, amps_x, phases_x, amps_y, phases_y

def main():
    parser = argparse.ArgumentParser(description="Generate 3D LHS Dataset for Inverse Design")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Total number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Number of samples to save per .pt file")
    parser.add_argument('--order_N', type=int, default=10,
                        help="Diffraction order X")
    parser.add_argument('--order_N_y', type=int, default=10,
                        help="Diffraction order Y")
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
    
    args = parser.parse_args()
    
    out_dir = os.path.join(project_root, 'Data', 'LHS_3D_Dataset')
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate LHS
    h_arr, inc_ang_arr, azi_ang_arr, wl_arr, amps_x_arr, phases_x_arr, amps_y_arr, phases_y_arr = get_lhs_samples(args.num_samples)
    
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
    
    print(f"Starting 3D LHS dataset generation: {args.num_samples} samples in {num_batches} batches.")
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
        # For 3D single-wavelength: absorptance is shape (1, 2) per sample
        all_wavelength = torch.zeros(batch_len)
        all_h = torch.zeros(batch_len)
        all_inc_ang = torch.zeros(batch_len)
        all_azi_ang = torch.zeros(batch_len)
        all_params_x = torch.zeros(batch_len, 5, 2)
        all_params_y = torch.zeros(batch_len, 5, 2)
        all_A_film_normal = torch.zeros(batch_len, 1, 2)
        all_A_grating_normal = torch.zeros(batch_len, 1, 2)
        all_A_film_oblique = torch.zeros(batch_len, 1, 2)
        all_A_grating_oblique = torch.zeros(batch_len, 1, 2)
        
        print(f"\nComputing Batch {batch_idx:04d} (Samples {start_idx} to {end_idx-1})...")
        sys.stdout.flush()
        for local_i, i in enumerate(tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx:04d}", mininterval=2.0, file=sys.stdout)):
            h = h_arr[i]
            inc_ang_deg = inc_ang_arr[i]
            azi_ang_deg = azi_ang_arr[i]
            wl = wl_arr[i]
            
            # Format params_x and params_y as (5, 2) tensors
            params_x_data = [[amps_x_arr[i, j], phases_x_arr[i, j]] for j in range(5)]
            params_y_data = [[amps_y_arr[i, j], phases_y_arr[i, j]] for j in range(5)]
            params_x = torch.tensor(params_x_data, dtype=geo_dtype, device=default_device)
            params_y = torch.tensor(params_y_data, dtype=geo_dtype, device=default_device)
            
            # Create a single wavelength tensor for this sample
            wavelength_tensor = torch.tensor([wl], dtype=torch.float64) + 1e-3
            
            base_config = RCWAConfig(
                grating_period=args.grating_period, grating_period_y=args.grating_period_y,
                h=float(h), order_N=args.order_N, order_N_y=args.order_N_y,
                n_layers=args.num_layers, height_per_layer=args.height_per_layer,
                nx=args.nx, ny=args.ny,
                add_reflector=not args.no_reflector, reflector_type=args.reflector_type,
                subpixel=not args.no_subpixel, grating_material=args.grating_material
            )
            
            # 1. Normal Incidence Calculation
            base_config.inc_ang = 1e-3 * np.pi/180
            base_config.azi_ang = 1e-3 * np.pi/180
            A_film_norm, A_grat_norm = get_absorptance_curve(
                params_x=params_x, params_y=params_y,
                wavelengths=wavelength_tensor, config=base_config
            )
            
            # 2. Oblique Incidence Calculation
            base_config.inc_ang = (inc_ang_deg + 1e-3) * np.pi/180
            base_config.azi_ang = (azi_ang_deg + 1e-3) * np.pi/180
            A_film_obl, A_grat_obl = get_absorptance_curve(
                params_x=params_x, params_y=params_y,
                wavelengths=wavelength_tensor, config=base_config
            )
            
            # Store into pre-allocated tensors
            all_wavelength[local_i] = float(wl)
            all_h[local_i] = float(h)
            all_inc_ang[local_i] = float(inc_ang_deg)
            all_azi_ang[local_i] = float(azi_ang_deg)
            all_params_x[local_i] = params_x.cpu()
            all_params_y[local_i] = params_y.cpu()
            all_A_film_normal[local_i] = A_film_norm.cpu()
            all_A_grating_normal[local_i] = A_grat_norm.cpu()
            all_A_film_oblique[local_i] = A_film_obl.cpu()
            all_A_grating_oblique[local_i] = A_grat_obl.cpu()
            
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
