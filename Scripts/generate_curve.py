import argparse
import itertools
import datetime
import os
import sys
import numpy as np
from dataclasses import asdict

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from Utils.utils import get_absorptance_curve, geo_dtype, RCWAConfig
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_tensor_arg(arg_str):
    """
    Parses a string like "40,0;10,3.14" into a torch tensor [[40.0, 0.0], [10.0, 3.14]]
    """
    if not arg_str:
        return None
    rows = arg_str.split(';')
    data = []
    for r in rows:
        data.append([float(x) for x in r.split(',')])
    return torch.tensor(data, dtype=geo_dtype, device=default_device)

def _process_curve_chunk(args_tuple):
    wavelength_chunk, params_x, params_y, config_dict = args_tuple
    import torch
    torch.set_num_threads(1)
    config = RCWAConfig(**config_dict)
    # We call the unmodified utils function, but just on a small chunk of wavelengths!
    A_film, A_grating = get_absorptance_curve(params_x=params_x, params_y=params_y, wavelengths=wavelength_chunk, config=config, show_progress=False)
    return A_film.cpu(), A_grating.cpu()

def main():
    parser = argparse.ArgumentParser(description="Generate absorptance curves for inverse design")
    parser.add_argument('--name', type=str, required=True, help="Base name for the output file (e.g., 'convergence_test')")
    parser.add_argument('--params_x', type=str, required=True, help="Params X e.g. '40,0;10,3.14'")
    parser.add_argument('--params_y', type=str, default="", help="Params Y e.g. '40,0' (leave empty for 2D)")
    parser.add_argument('--order_N', type=int, nargs='+', default=[10], help="List of X diffraction orders")
    parser.add_argument('--order_N_y', type=int, nargs='+', default=None, help="List of Y diffraction orders")
    parser.add_argument('--num_layers', type=int, nargs='+', default=[10], help="List of layer counts for staircase approximation")
    parser.add_argument('--height_per_layer', type=float, default=None, help="Overrides num_layers to fix grating resolution")
    parser.add_argument('--wavelengths', type=float, nargs=3, default=[300, 1100, 1601], help="Wavelengths: start end steps")
    parser.add_argument('--nx', type=int, default=20000, help="Grid size nx")
    parser.add_argument('--ny', type=int, default=1, help="Grid size ny (use 1 for 2D, scale up for 3D)")
    parser.add_argument('--grating_period', type=float, default=1000.0, help="Grating period X (nm)")
    parser.add_argument('--grating_period_y', type=float, default=None, help="Grating period Y (nm)")
    parser.add_argument('--h', type=float, default=1000.0, help="Thickness h (nm)")
    parser.add_argument('--inc_ang', type=float, default=30.0, help="Incident angle in degrees")
    parser.add_argument('--azi_ang', type=float, default=0.0, help="Azimuthal angle in degrees")
    parser.add_argument('--grating_material', type=str, default='Si', help="Material for the grating layer (e.g. Si, TiO2, Si3N4)")
    parser.add_argument('--no_reflector', action='store_true', help="Disable the bottom reflector")
    parser.add_argument('--reflector_type', type=str, default='pec', help="Reflector type (e.g., pec, Ag)")
    parser.add_argument('--no_subpixel', action='store_true', help="Disable subpixel smoothing")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of CPU cores for parallel wavelength chunking")
    
    args = parser.parse_args()
    
    params_x = parse_tensor_arg(args.params_x)
    params_y = parse_tensor_arg(args.params_y)
    
    # Generate wavelengths tensor (offset by 1e-3 for stability as in the notebook)
    wl_start, wl_end, wl_steps = args.wavelengths
    wavelengths = torch.linspace(wl_start, wl_end, int(wl_steps), dtype=torch.float64) + 1e-3
    
    # Convert angles to radians and apply stability offset
    inc_ang_rad = (args.inc_ang + 1e-3) * (np.pi / 180)
    azi_ang_rad = (args.azi_ang + 1e-3) * (np.pi / 180)
    
    results_dict = {}
    
    # Create base config to save in metadata
    base_config = RCWAConfig(
        inc_ang=inc_ang_rad, azi_ang=azi_ang_rad,
        grating_period=args.grating_period, grating_period_y=args.grating_period_y,
        h=args.h, nx=args.nx, ny=args.ny,
        height_per_layer=args.height_per_layer,
        add_reflector=not args.no_reflector, reflector_type=args.reflector_type, 
        subpixel=not args.no_subpixel, grating_material=args.grating_material
    )
    
    if args.n_jobs > 1:
        import multiprocessing as mp
        num_chunks = min(args.n_jobs * 10, len(wavelengths))
        print(f"Parallelizing {len(wavelengths)} wavelengths across {num_chunks} chunks ({args.n_jobs} workers)...")
        chunks = torch.tensor_split(wavelengths, num_chunks)
        pool = mp.Pool(processes=args.n_jobs)
    else:
        pool = None
    
    # Evaluate combinations. If order_N_y isn't given, assume symmetric orders (o_y = o_x)
    from tqdm import tqdm
    if args.order_N_y is None:
        combinations = list(itertools.product(args.order_N, args.num_layers))
        print(f"Starting batch generation for {len(combinations)} parameter combination(s)...")
        
        for (o_x, n_layers) in tqdm(combinations, desc="Combinations", file=sys.stdout, mininterval=2.0):
            o_y = o_x  # Symmetric
            key = f"order_x_{o_x}_order_y_{o_y}_layers_{n_layers}"
            print(f"\nRunning -> {key}")
            sys.stdout.flush()
            
            config = RCWAConfig(
                inc_ang=inc_ang_rad, azi_ang=azi_ang_rad,
                grating_period=args.grating_period, grating_period_y=args.grating_period_y,
                h=args.h, order_N=o_x, order_N_y=o_y, nx=args.nx, ny=args.ny,
                n_layers=n_layers, height_per_layer=args.height_per_layer, add_reflector=not args.no_reflector, reflector_type=args.reflector_type, 
                subpixel=not args.no_subpixel, grating_material=args.grating_material
            )
            
            if pool:
                config_dict = asdict(config)
                tasks = [(c, params_x, params_y, config_dict) for c in chunks]
                results = list(tqdm(pool.imap(_process_curve_chunk, tasks), total=len(tasks), desc="Wavelength Chunks", file=sys.stdout, mininterval=2.0, leave=False))
                A_film = torch.cat([r[0] for r in results], dim=0)
                A_grating = torch.cat([r[1] for r in results], dim=0)
                results_dict[key] = {'A_film': A_film, 'A_grating': A_grating}
            else:
                A_film, A_grating = get_absorptance_curve(params_x=params_x, params_y=params_y, wavelengths=wavelengths, config=config, show_progress=True)
                results_dict[key] = {'A_film': A_film.cpu(), 'A_grating': A_grating.cpu()}
                
    else:
        combinations = list(itertools.product(args.order_N, args.order_N_y, args.num_layers))
        print(f"Starting batch generation for {len(combinations)} parameter combination(s)...")
        
        for (o_x, o_y, n_layers) in tqdm(combinations, desc="Combinations", file=sys.stdout, mininterval=2.0):
            key = f"order_x_{o_x}_order_y_{o_y}_layers_{n_layers}"
            print(f"\nRunning -> {key}")
            sys.stdout.flush()
            
            config = RCWAConfig(
                inc_ang=inc_ang_rad, azi_ang=azi_ang_rad,
                grating_period=args.grating_period, grating_period_y=args.grating_period_y,
                h=args.h, order_N=o_x, order_N_y=o_y, nx=args.nx, ny=args.ny,
                n_layers=n_layers, height_per_layer=args.height_per_layer, add_reflector=not args.no_reflector, reflector_type=args.reflector_type, 
                subpixel=not args.no_subpixel, grating_material=args.grating_material
            )
            
            if pool:
                config_dict = asdict(config)
                tasks = [(c, params_x, params_y, config_dict) for c in chunks]
                results = list(tqdm(pool.imap(_process_curve_chunk, tasks), total=len(tasks), desc="Wavelength Chunks", file=sys.stdout, mininterval=2.0, leave=False))
                A_film = torch.cat([r[0] for r in results], dim=0)
                A_grating = torch.cat([r[1] for r in results], dim=0)
                results_dict[key] = {'A_film': A_film, 'A_grating': A_grating}
            else:
                A_film, A_grating = get_absorptance_curve(params_x=params_x, params_y=params_y, wavelengths=wavelengths, config=config, show_progress=True)
                results_dict[key] = {'A_film': A_film.cpu(), 'A_grating': A_grating.cpu()}
                
    if pool:
        pool.close()
        pool.join()
        
    # Bundle results and all simulation metadata for plotting later
    save_data = {
        'results': results_dict,
        'metadata': {
            'params_x': params_x.cpu() if params_x is not None else None,
            'params_y': params_y.cpu() if params_y is not None else None,
            'wavelengths': wavelengths.cpu(),
            'config': asdict(base_config)
        }
    }
    
    # Create structured data directory
    out_dir = os.path.join(project_root, 'Data', 'AbsorptanceCurves')
    os.makedirs(out_dir, exist_ok=True)
    
    # Append date to user's requested base name
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"{args.name}_{date_str}.pt"
    filepath = os.path.join(out_dir, filename)
    
    torch.save(save_data, filepath)
    print(f"\nSUCCESS: Data saved to {filepath}")

if __name__ == "__main__":
    main()
