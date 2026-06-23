import time
import math
import torch
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utils.utils import get_absorptance, RCWAConfig, get_continuous_boundary, get_staircase_sine_eps, get_material_eps, si_eps, ag_eps, default_device, get_incident_power
import torcwa

def get_absorptance_fast(params_x, params_y, wavelength, config: RCWAConfig):
    # This is a stripped down version of get_absorptance that only solves the global S-matrix
    # and completely skips the slow field_xy reconstruction.
    device = default_device
    sim_dtype = torch.complex64

    # Setup config
    inc_ang, azi_ang = config.inc_ang, config.azi_ang
    grating_period, grating_period_y, h = config.grating_period, config.grating_period_y, config.h
    n_layers, height_per_layer = config.n_layers, config.height_per_layer
    order_N, order_N_y = config.order_N, config.order_N_y
    grating_material = config.grating_material
    subpixel = config.subpixel
    nx = config.nx
    is_3d = False
    
    grating_period_y = grating_period_y or grating_period
    order_N_y = 0
    ny = 1
    add_reflector = config.add_reflector
    reflector_type = config.reflector_type

    torcwa.rcwa_geo.nx = nx
    torcwa.rcwa_geo.ny = ny
    torcwa.rcwa_geo.grid()
    L = [grating_period, grating_period_y]
    order = [order_N, order_N_y]

    _, grating_height = get_continuous_boundary(torcwa.rcwa_geo.x, params_x, grating_period)
    effective_n_layers = max(1, math.ceil(grating_height / height_per_layer)) if height_per_layer else n_layers

    sine_eps = get_staircase_sine_eps(
        x=torcwa.rcwa_geo.x, params_x=params_x, grating_period=grating_period,
        num_layers=effective_n_layers, eps_high=get_material_eps(grating_material, wavelength), subpixel=subpixel if effective_n_layers>1 else True
    )
        
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device, avoid_Pinv_instability=True)
    sim.add_input_layer()
    if add_reflector:
        reflector_eps = torch.tensor(-10000.0 + 0.0j, dtype=sim_dtype, device=device) if reflector_type == 'pec' else ag_eps(wavelength)
        sim.add_output_layer(eps=reflector_eps)
    else:
        sim.add_output_layer()

    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    for i in range(effective_n_layers):
        eps_slice = sine_eps[..., -1-i] + 1e-10j
        if not is_3d:
            eps_slice = eps_slice.unsqueeze(-1)
        sim.add_layer(thickness=grating_height/effective_n_layers, eps=eps_slice)
        
    sim.add_layer(thickness=h, eps=si_eps(wavelength))
    
    sim.solve_global_smatrix()
    return sim

def test_rcwa_speed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to get exact config
    batch_file = PROJECT_ROOT / "Data" / "LHS_Dataset_Si" / "batch_0000.pt"
    data = torch.load(batch_file, map_location=device, weights_only=False)
    
    # 1. Setup a real geometry from the dataset
    params_x = data["params_x"][0] # Wait, the key is params_x not geometries
    h_val = data["h"][0].item()
    
    config_dict = data["metadata"]["config"]
    config_dict["h"] = h_val
    config = RCWAConfig(**config_dict)
    
    wavelengths = data["metadata"]["wavelengths"].to(device)
    
    print(f"Loaded config: order_N={config.order_N}, height_per_layer={config.height_per_layer}, h={h_val}")
    
    # We will run both methods inside a loop across wavelengths, exactly like get_absorptance_curve does.
    A_total_orig_p = []
    A_total_orig_s = []
    
    print(f"--- Method 1: Original field_xy() method (161 Wavelengths) ---")
    t0 = time.time()
    for wl in wavelengths:
        res_orig = get_absorptance(params_x=params_x, params_y=None, wavelength=wl, config=config)
        A_total_orig_p.append(res_orig[2][0].item() + res_orig[3][0].item())
        A_total_orig_s.append(res_orig[2][1].item() + res_orig[3][1].item())
    t1 = time.time()
    time_orig = t1 - t0
    print(f"Time (field_xy): {time_orig:.4f} seconds")
    
    A_total_fast_p = []
    A_total_fast_s = []

    print("\n--- Method 2: S_parameters() method (161 Wavelengths) ---")
    t0 = time.time()
    
    # Same orders as the dataset (N=5)
    n_orders = 2 * config.order_N + 1
    orders_1d = torch.arange(-config.order_N, config.order_N + 1, device=device)
    orders = torch.stack([orders_1d, torch.zeros_like(orders_1d)], dim=-1) # (N, 2)
    
    for wl in wavelengths:
        # Use our fast initialization that completely skips field_xy
        sim = get_absorptance_fast(params_x=params_x, params_y=None, wavelength=wl, config=config)
        
        # Calculate R and T for P-pol (using ps-notation: 'pp' + 'sp' for P-pol incident)
        S_tp_p = sim.S_parameters(orders, direction='f', port='t', polarization='pp', power_norm=True)
        S_ts_p = sim.S_parameters(orders, direction='f', port='t', polarization='sp', power_norm=True)
        T_p = torch.sum(torch.abs(S_tp_p)**2 + torch.abs(S_ts_p)**2)
        
        S_rp_p = sim.S_parameters(orders, direction='f', port='r', polarization='pp', power_norm=True)
        S_rs_p = sim.S_parameters(orders, direction='f', port='r', polarization='sp', power_norm=True)
        R_p = torch.sum(torch.abs(S_rp_p)**2 + torch.abs(S_rs_p)**2)
        
        # Calculate R and T for S-pol
        S_tp_s = sim.S_parameters(orders, direction='f', port='t', polarization='ps', power_norm=True)
        S_ts_s = sim.S_parameters(orders, direction='f', port='t', polarization='ss', power_norm=True)
        T_s = torch.sum(torch.abs(S_tp_s)**2 + torch.abs(S_ts_s)**2)
        
        S_rp_s = sim.S_parameters(orders, direction='f', port='r', polarization='ps', power_norm=True)
        S_rs_s = sim.S_parameters(orders, direction='f', port='r', polarization='ss', power_norm=True)
        R_s = torch.sum(torch.abs(S_rp_s)**2 + torch.abs(S_rs_s)**2)
        
        A_total_fast_p.append(1.0 - R_p.item() - T_p.item())
        A_total_fast_s.append(1.0 - R_s.item() - T_s.item())
        
    t1 = time.time()
    time_fast = t1 - t0
    print(f"Time (S_params + sim setup): {time_fast:.4f} seconds")
    
    A_orig_p = np.array(A_total_orig_p)
    A_fast_p = np.array(A_total_fast_p)
    A_orig_s = np.array(A_total_orig_s)
    A_fast_s = np.array(A_total_fast_s)
    
    mae_p = np.mean(np.abs(A_orig_p - A_fast_p))
    mae_s = np.mean(np.abs(A_orig_s - A_fast_s))
    
    print(f"\n--- Comparison ---")
    print(f"Speedup multiplier: {time_orig / time_fast:.2f}x")
    print(f"Curve MAE (P-Pol): {mae_p:.6e}")
    print(f"Curve MAE (S-Pol): {mae_s:.6e}")
    
if __name__ == "__main__":
    test_rcwa_speed()
