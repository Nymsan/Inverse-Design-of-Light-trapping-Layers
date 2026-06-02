import argparse
import itertools
import datetime
import os
import torch
import sys
import numpy as np

# Ensure project root is in path so we can import Utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.utils import get_absorptance_curve, geo_dtype, device, RCWAConfig

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
    return torch.tensor(data, dtype=geo_dtype, device=device)

def main():
    parser = argparse.ArgumentParser(description="Generate absorptance curves for inverse design")
    parser.add_argument('--name', type=str, required=True, help="Base name for the output file (e.g., 'convergence_test')")
    parser.add_argument('--params_x', type=str, required=True, help="Params X e.g. '40,0;10,3.14'")
    parser.add_argument('--params_y', type=str, default="", help="Params Y e.g. '40,0' (leave empty for 2D)")
    parser.add_argument('--order_N', type=int, nargs='+', default=[15], help="List of X diffraction orders")
    parser.add_argument('--order_N_y', type=int, nargs='+', default=None, help="List of Y diffraction orders")
    parser.add_argument('--num_layers', type=int, nargs='+', default=[10], help="List of layer counts for staircase approximation")
    parser.add_argument('--wavelengths', type=float, nargs=3, default=[300, 1100, 1601], help="Wavelengths: start end steps")
    parser.add_argument('--nx', type=int, default=20000, help="Grid size nx")
    parser.add_argument('--ny', type=int, default=1, help="Grid size ny (use 1 for 2D, scale up for 3D)")
    parser.add_argument('--grating_period', type=float, default=1000.0, help="Grating period X (nm)")
    parser.add_argument('--grating_period_y', type=float, default=None, help="Grating period Y (nm)")
    parser.add_argument('--h', type=float, default=1000.0, help="Thickness h (nm)")
    parser.add_argument('--inc_ang', type=float, default=30.0, help="Incident angle in degrees")
    parser.add_argument('--azi_ang', type=float, default=0.0, help="Azimuthal angle in degrees")
    
    args = parser.parse_args()
    
    params_x = parse_tensor_arg(args.params_x)
    params_y = parse_tensor_arg(args.params_y)
    
    # Generate wavelengths tensor (offset by 1e-3 for stability as in the notebook)
    wl_start, wl_end, wl_steps = args.wavelengths
    wavelengths = torch.linspace(wl_start, wl_end, int(wl_steps), dtype=torch.float64) + 1e-3
    
    # Convert angles to radians and apply stability offset
    inc_ang_rad = (args.inc_ang + 1e-3) * (np.pi / 180)
    azi_ang_rad = (args.azi_ang + 1e-3) * (np.pi / 180)
    
    order_N_y_list = args.order_N_y if args.order_N_y is not None else [None]
    
    results_dict = {}
    
    # Evaluate combinations. If order_N_y isn't given, assume symmetric orders (o_y = o_x)
    if args.order_N_y is None:
        combinations = list(itertools.product(args.order_N, args.num_layers))
        print(f"Starting batch generation for {len(combinations)} parameter combination(s)...")
        
        for (o_x, n_layers) in combinations:
            o_y = o_x  # Symmetric
            key = f"order_x_{o_x}_order_y_{o_y}_layers_{n_layers}"
            print(f"Running -> {key}")
            
            config = RCWAConfig(
                inc_ang=inc_ang_rad, azi_ang=azi_ang_rad,
                grating_period=args.grating_period, grating_period_y=args.grating_period_y,
                h=args.h, order_N=o_x, order_N_y=o_y, nx=args.nx, ny=args.ny,
                n_layers=n_layers, add_reflector=True, reflector_type='pec', subpixel=True
            )
            A_film, A_grating = get_absorptance_curve(params_x=params_x, params_y=params_y, wavelengths=wavelengths, config=config)
            
            results_dict[key] = {'A_film': A_film.cpu(), 'A_grating': A_grating.cpu()}
    else:
        combinations = list(itertools.product(args.order_N, args.order_N_y, args.num_layers))
        print(f"Starting batch generation for {len(combinations)} parameter combination(s)...")
        
        for (o_x, o_y, n_layers) in combinations:
            key = f"order_x_{o_x}_order_y_{o_y}_layers_{n_layers}"
            print(f"Running -> {key}")
            
            config = RCWAConfig(
                inc_ang=inc_ang_rad, azi_ang=azi_ang_rad,
                grating_period=args.grating_period, grating_period_y=args.grating_period_y,
                h=args.h, order_N=o_x, order_N_y=o_y, nx=args.nx, ny=args.ny,
                n_layers=n_layers, add_reflector=True, reflector_type='pec', subpixel=True
            )
            A_film, A_grating = get_absorptance_curve(params_x=params_x, params_y=params_y, wavelengths=wavelengths, config=config)
            
            results_dict[key] = {'A_film': A_film.cpu(), 'A_grating': A_grating.cpu()}
        
    # Bundle results and all simulation metadata for plotting later
    save_data = {
        'results': results_dict,
        'metadata': {
            'params_x': params_x.cpu() if params_x is not None else None,
            'params_y': params_y.cpu() if params_y is not None else None,
            'wavelengths': wavelengths.cpu(),
            'inc_ang_deg': args.inc_ang,
            'azi_ang_deg': args.azi_ang,
            'grating_period': args.grating_period,
            'grating_period_y': args.grating_period_y,
            'h': args.h,
            'nx': args.nx,
            'ny': args.ny
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
