import sys
import contextlib
import numpy as np
import torch
from matplotlib import pyplot as plt
import torcwa
from pvlib import spectrum
from refractiveindex import RefractiveIndexMaterial
from tqdm import tqdm
import math
from typing import Optional
from dataclasses import dataclass  # noqa: E402
from pathlib import Path

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class RCWAConfig:
    grating_period: float = 1000.0
    grating_period_y: Optional[float] = None
    h: float = 1000.0
    order_N: int = 15
    order_N_y: Optional[int] = None
    nx: int = 1000
    ny: int = 1
    n_layers: int = 10
    height_per_layer: Optional[float] = None
    subpixel: bool = True
    add_reflector: bool = True
    reflector_type: str = 'pec'
    inc_ang: float = 0.0
    azi_ang: float = 0.0
    grating_material: str = 'Si'

def split_params_3d(params):
    """Split a combined params tensor of shape (2, n, 2) or a tuple into params_x and params_y."""
    if isinstance(params, (list, tuple)) and len(params) == 2:
        return params[0], params[1]
    if isinstance(params, torch.Tensor) and params.dim() == 3 and params.shape[0] == 2:
        return params[0], params[1]
    return params, None

def _compute_1d_profile(r, params, grating_period):
    """Compute 1D grating profile. Returns height_profile, peak_to_peak_height."""
    grating_height = 2 * torch.sum(params[:, 0]) + 1e-9
    freqs = torch.arange(1, params.shape[0]+1, dtype=r.dtype, device=r.device).unsqueeze(1)
    cosines = torch.cos(2. * np.pi * freqs * (r.unsqueeze(0) / grating_period) - params[:, 1].unsqueeze(1))
    cosines = cosines * params[:, 0].unsqueeze(1)
    profile = grating_height / 2 + torch.sum(cosines, dim=0)
    return profile, grating_height

def get_staircase_sine_eps(x, params_x, grating_period, num_layers, eps_high, eps_low=1., subpixel=True, y=None, params_y=None, grating_period_y=None):
    """Generate sine grating in staircase approximation. Supports 1D and 2D gratings."""
    params_x, params_y_split = split_params_3d(params_x)
    params_y = params_y if params_y is not None else params_y_split
    grating_period_y = grating_period_y or grating_period
    
    profile_x, height_x = _compute_1d_profile(x, params_x, grating_period)
    
    if params_y is not None and y is not None:
        profile_y, height_y = _compute_1d_profile(y, params_y, grating_period_y)
        eps_profile = profile_x.unsqueeze(1) + profile_y.unsqueeze(0)
        grating_height = height_x + height_y
        nx, ny = x.shape[0], y.shape[0]
        thresholds = torch.arange(0, num_layers, device=x.device) * (grating_height / num_layers)
        thresholds = thresholds.unsqueeze(0).unsqueeze(0).expand(nx, ny, -1)
        eps_profile = eps_profile.unsqueeze(-1).expand(-1, -1, num_layers)
    else:
        eps_profile = profile_x
        grating_height = height_x
        nx = x.shape[0]
        thresholds = torch.arange(0, num_layers, device=x.device) * (grating_height / num_layers)
        thresholds = thresholds.unsqueeze(0).expand(nx, -1)
        eps_profile = eps_profile.unsqueeze(1).expand(-1, num_layers)

    cell_height = grating_height / num_layers
    eps = torch.clamp((eps_profile - thresholds) / cell_height, min=0, max=1)
    
    if not subpixel:
        # The Straight-Through Estimator (STE) trick
        # This avoids killing gradients
        eps = (torch.round(eps) - eps).detach() + eps
        
    return eps_low + (eps_high - eps_low) * eps

# light
spectra = spectrum.get_reference_spectra()
am15g = spectra['global']
def sun_weights(w):
    device = w.device if hasattr(w, "device") else default_device
    vals = am15g[w.cpu().numpy()]
    if hasattr(vals, "to_numpy"):
        vals = vals.to_numpy()
    return torch.tensor(vals, dtype=geo_dtype, device=device)

def get_jsc_scaling_factor(wl_len):
    """
    Returns the scaling factor to convert the sum(A * S * lambda) into Jsc [mA/cm^2].
    Assumes wavelengths span 300 to 1100 nm.
    Constants:
    q = 1.602176634e-19 C
    h = 6.62607015e-34 J s
    c = 299792458.0 m/s
    """
    d_lambda = (1100.0 - 300.0) / (wl_len - 1)
    # Jsc [A/m^2] = sum(A * S * d_lambda * lambda * 1e-9 * q / (h * c))
    # Jsc [mA/cm^2] = Jsc [A/m^2] * 0.1
    q = 1.602176634e-19
    h = 6.62607015e-34
    c = 299792458.0
    factor = 1e-9 * q / (h * c) * 0.1 * d_lambda
    return factor

# material
_materials_db = {
    'Si': RefractiveIndexMaterial(shelf='main', book='Si', page='Green-2008'),
    'Ag': RefractiveIndexMaterial(shelf='main', book='Ag', page='McPeak'),
    'TiO2': RefractiveIndexMaterial(shelf='main', book='TiO2', page='Siefke'),
    'Si3N4': RefractiveIndexMaterial(shelf='main', book='Si3N4', page='Luke')
}

def get_material_eps(material_name, w):
    device = w.device if hasattr(w, "device") else default_device
    if material_name not in _materials_db:
        raise ValueError(f"Material {material_name} not found in library. Available: {list(_materials_db.keys())}")
    mat = _materials_db[material_name]
    
    w_val = w.item() if isinstance(w, torch.Tensor) else float(w)
        
    n = mat.get_refractive_index(w_val)
    
    try:
        k = mat.get_extinction_coefficient(w_val)
        if k is None:
            k = 0.0
    except Exception:
        k = 0.0
        
    return torch.tensor(n + 1j * k, dtype=sim_dtype, device=device)**2

def si_eps(w):
    return get_material_eps('Si', w)

def ag_eps(w):
    return get_material_eps('Ag', w)

def get_continuous_boundary(x, params_x, grating_period, y=None, params_y=None, grating_period_y=None):
    """Extracts the continuous wavy boundary. Returns the boundary and the grating height."""
    params_x, params_y_split = split_params_3d(params_x)
    params_y = params_y if params_y is not None else params_y_split
    grating_period_y = grating_period_y or grating_period
    
    profile_x, height_x = _compute_1d_profile(x, params_x, grating_period)
    
    if params_y is not None and y is not None:
        profile_y, height_y = _compute_1d_profile(y, params_y, grating_period_y)
        original_profile = profile_x.unsqueeze(1) + profile_y.unsqueeze(0)
        grating_height = height_x + height_y
    else:
        original_profile = profile_x
        grating_height = height_x
        
    z_boundary = grating_height - original_profile

    return z_boundary, grating_height.item()
    
def get_incident_power(pol,wavelength,inc_ang,azi_ang,grating_period,order,grating_period_y=None):
    device = wavelength.device if hasattr(wavelength, 'device') else default_device
    is_3d = order[1] > 0
    grating_period_y = grating_period_y or grating_period
    L = [grating_period, grating_period_y]
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device, avoid_Pinv_instability=True)
    sim.add_input_layer()
    sim.add_output_layer()
    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=pol, direction='forward', notation='ps')
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torch.tensor([0],device=device), torch.tensor([0]), y=0.0)
    P_inc = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))

    if is_3d:
        return P_inc.item() * grating_period * grating_period_y
    return P_inc.item() * grating_period

def get_absorptance(params_x, params_y, wavelength, config: RCWAConfig):
    device = params_x.device if hasattr(params_x, "device") else default_device
    inc_ang, azi_ang = config.inc_ang, config.azi_ang
    grating_period, grating_period_y, h = config.grating_period, config.grating_period_y, config.h
    order_N, order_N_y = config.order_N, config.order_N_y
    nx, ny, n_layers, height_per_layer, subpixel = config.nx, config.ny, config.n_layers, config.height_per_layer, config.subpixel
    add_reflector, reflector_type = config.add_reflector, config.reflector_type
    grating_material = config.grating_material
    if not isinstance(wavelength, torch.Tensor):
        wavelength = torch.tensor(wavelength, dtype=geo_dtype)
    params_x, params_y_split = split_params_3d(params_x)
    params_y = params_y if params_y is not None else params_y_split
    is_3d = params_y is not None
    
    grating_period_y = grating_period_y or grating_period
    order_N_y = order_N_y if order_N_y is not None else (order_N if is_3d else 0)
    
    if not is_3d:
        order_N_y = 0
        ny = 1
    elif is_3d and ny == 1:
        ny = nx  # scale ny if not explicitly provided
        
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = grating_period
    torcwa.rcwa_geo.Ly = grating_period_y
    torcwa.rcwa_geo.nx = nx
    torcwa.rcwa_geo.ny = ny
    torcwa.rcwa_geo.grid()
    L = [grating_period, grating_period_y]
    
    order = [order_N, order_N_y]

    _, grating_height = get_continuous_boundary(
        torcwa.rcwa_geo.x, params_x, grating_period,
        y=torcwa.rcwa_geo.y if is_3d else None,
        params_y=params_y, grating_period_y=grating_period_y
    )

    effective_n_layers = n_layers
    if height_per_layer is not None:
        effective_n_layers = max(1, math.ceil(grating_height / height_per_layer))

    sine_eps = get_staircase_sine_eps(
        x=torcwa.rcwa_geo.x, params_x=params_x, grating_period=grating_period,
        num_layers=effective_n_layers, eps_high=get_material_eps(grating_material, wavelength), subpixel=subpixel if effective_n_layers>1 else True,
        y=torcwa.rcwa_geo.y if is_3d else None, params_y=params_y, grating_period_y=grating_period_y
    )
        
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device, avoid_Pinv_instability=True)
    sim.add_input_layer()
    if add_reflector:
        if reflector_type == 'pec':
            reflector_eps = torch.tensor(-10000.0 + 0.0j, dtype=sim_dtype, device=device)

        elif reflector_type == 'Ag':
            reflector_eps = ag_eps(wavelength)

        sim.add_output_layer(eps=reflector_eps)
    else:
        sim.add_output_layer()

    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    
    for i in range(effective_n_layers):
        eps_slice = sine_eps[..., -1-i] + 1e-10j # This was added after the latest dataset generation 15.06.26, but maybe it is smart
        if not is_3d:
            eps_slice = eps_slice.unsqueeze(-1)
        sim.add_layer(thickness=grating_height/effective_n_layers, eps=eps_slice)
        
    sim.add_layer(thickness=h, eps=si_eps(wavelength))
    

    sim.solve_global_smatrix()
    
    results = {}
    for pol_idx, pol in enumerate([[1., 0.], [0., 1.]]): #first result is p-pol, second s-pol
        P_inc = get_incident_power(pol=pol,wavelength=wavelength,inc_ang=inc_ang,azi_ang=azi_ang,grating_period=grating_period,order=order,grating_period_y=grating_period_y)
        sim.source_planewave(amplitude=pol, direction='forward', notation='ps')
        
        area = grating_period * grating_period_y if is_3d else grating_period
        
        #Note the layer num argument of .field_xy is a bit weird here. Torcwa.rcwa source code hardcodes -1 to be the input layer
        #and hardcodes Layer_n (here = effective_n_layers+1) to be the output space.
        
        # z_air (incident space, layer_num=-1)
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xy(-1, torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, z_prop=-5*h)
        S_z_air = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
        P_air = torch.mean(S_z_air) * area
        
        # z_top (top of bulk layer = layer num_layers, z_prop=0)
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xy(effective_n_layers, torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, z_prop=0.0)
        S_z_top = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
        P_top = torch.mean(S_z_top) * area
        
        # z_bot (top of output space = layer num_layers+1, z_prop=0)
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xy(effective_n_layers+1, torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, z_prop=0.0)
        S_z_bot = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
        P_bot = torch.mean(S_z_bot) * area
        
        #TODO Add a sanity check using a volume integral for the power.
        
        results[pol_idx] = {
            'A_film': (P_top - P_bot) / P_inc,
            'A_grating': (P_air - P_top) / P_inc,
            'R': (P_inc - P_air) / P_inc,
            'T': P_bot / P_inc,
            'P_abs_film': P_top - P_bot,
            'P_abs_grating': P_air - P_top,
            #'P_slices': torch.stack([P_inc,P_top, P_bot, P_air])
        }

    A_film = torch.stack([results[0]['A_film'], results[1]['A_film']])
    A_grating = torch.stack([results[0]['A_grating'], results[1]['A_grating']])
    Reflectance = torch.stack([results[0]['R'], results[1]['R']])
    Transmittance = torch.stack([results[0]['T'], results[1]['T']])
    P_abs_film = torch.stack([results[0]['P_abs_film'], results[1]['P_abs_film']])
    P_abs_grating = torch.stack([results[0]['P_abs_grating'], results[1]['P_abs_grating']])
    #P_slices = torch.stack([results[0]['P_slices'], results[1]['P_slices']])

    return sim, sine_eps, A_film, A_grating, Reflectance, Transmittance, P_abs_film, P_abs_grating#, P_slices

def get_absorptance_curve(params_x, params_y, wavelengths, config: RCWAConfig, show_progress=False, requires_grad=False):
    device = params_x.device if hasattr(params_x, "device") else default_device
    wavelengths = wavelengths.to(device)
    if params_y is not None:
        params_y = params_y.to(device)
    inc_ang, azi_ang = config.inc_ang, config.azi_ang
    grating_period, grating_period_y, h = config.grating_period, config.grating_period_y, config.h
    order_N, order_N_y = config.order_N, config.order_N_y
    nx, ny, n_layers, height_per_layer, subpixel = config.nx, config.ny, config.n_layers, config.height_per_layer, config.subpixel
    add_reflector, reflector_type = config.add_reflector, config.reflector_type
    
    A_film_list = []
    A_grating_list = []
    
    iterator = tqdm(wavelengths, leave=False, file=sys.stdout, mininterval=2.0) if show_progress else wavelengths
    context = contextlib.nullcontext() if requires_grad else torch.no_grad()
    
    for i, wavelength in enumerate(iterator):
        temp_config = RCWAConfig(
            grating_period=grating_period, grating_period_y=grating_period_y, h=h,
            order_N=order_N, order_N_y=order_N_y, nx=nx, ny=ny, n_layers=n_layers, height_per_layer=height_per_layer, subpixel=subpixel,
            add_reflector=add_reflector, reflector_type=reflector_type,
            inc_ang=inc_ang, azi_ang=azi_ang, grating_material=config.grating_material
        )
        with context:
            A_film, A_grating = get_absorptance(params_x=params_x, params_y=params_y, wavelength=wavelength, config=temp_config)[2:4]

            A_film_list.append(A_film if requires_grad else A_film.cpu())
            A_grating_list.append(A_grating if requires_grad else A_grating.cpu())
            if not requires_grad and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    Absorptances_film = torch.stack(A_film_list, dim=0)
    Absorptances_grating = torch.stack(A_grating_list, dim=0)
    return Absorptances_film, Absorptances_grating

    

def plot_fields(sim, x_plot, z_plot, wavelength, polarization, params_x, params_y, config: RCWAConfig, field=None, thickness=2, y_plot=None, slice_plane='xz', slice_val=0.0):
    """
    Plots fields for a chosen 2D slice plane ('xz', 'yz', or 'xy').
    If `field` is None, plots a 4x3 grid.
    If `field` is a string (e.g., 'Ex', 'Snorm'), plots only that field.
    """
    inc_ang, azi_ang = config.inc_ang, config.azi_ang
    grating_period, grating_period_y, h = config.grating_period, config.grating_period_y, config.h
    sim.source_planewave(amplitude=polarization, direction='forward', notation='ps')
    dev = sim._device
    x_plot = x_plot.to(dev) if x_plot is not None else None
    z_plot = z_plot.to(dev) if z_plot is not None else None
    if y_plot is not None:
        y_plot = y_plot.to(dev)

    # Determine slice fields and extents
    if slice_plane == 'xz':
        if x_plot is None or z_plot is None:
            raise ValueError("x_plot and z_plot must be provided for xz slice.")
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(x_plot, z_plot, y=torch.tensor(slice_val, device=dev))
        v1_cpu, v2_cpu = x_plot.cpu().numpy(), z_plot.cpu().numpy()
        xlabel, ylabel = 'x (nm)', 'z (nm)'
        title_base = f"xz-plane field distribution at y = {slice_val} nm"
        
    elif slice_plane == 'yz' or slice_plane == 'zy':
        if y_plot is None or z_plot is None:
            raise ValueError("y_plot and z_plot must be provided for yz slice.")
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_yz(y_plot, z_plot, x=torch.tensor(slice_val, device=dev))
        v1_cpu, v2_cpu = y_plot.cpu().numpy(), z_plot.cpu().numpy()
        xlabel, ylabel = 'y (nm)', 'z (nm)'
        title_base = f"yz-plane field distribution at x = {slice_val} nm"
        
    elif slice_plane == 'xy':
        if x_plot is None or y_plot is None:
            raise ValueError("x_plot and y_plot must be provided for xy slice.")
        
        # Map absolute z (slice_val) to torcwa layer_num and z_prop
        zm = [0.0]
        for L in sim.thickness:
            zm.append(zm[-1] + L)
            
        z = slice_val
        if z < 0:
            layer_num = -1
            z_prop = z
        elif z >= zm[-1]:
            layer_num = sim.layer_N
            z_prop = z - zm[-1]
        else:
            for l in range(len(zm)-1):
                if zm[l] <= z < zm[l+1]:
                    layer_num = l
                    z_prop = z - zm[l]
                    break
                    
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xy(layer_num, x_plot, y_plot, z_prop=z_prop)
        v1_cpu, v2_cpu = x_plot.cpu().numpy(), y_plot.cpu().numpy()
        xlabel, ylabel = 'x (nm)', 'y (nm)'
        title_base = f"xy-plane field distribution at z = {slice_val} nm"
        
    else:
        raise ValueError(f"Unknown slice_plane: {slice_plane}")

    extent = [v1_cpu[0], v1_cpu[-1], v2_cpu[0], v2_cpu[-1]]
    title_base += f"\n$\\lambda$ = {wavelength:.3f} nm, pol = {polarization} in ps-basis \ninc = {inc_ang*180/np.pi:.1f}°, azi = {azi_ang*180/np.pi:.1f}°"

    # Field magnitudes and Poynting vector
    Enorm = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
    Hnorm = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)
    Sx = 0.5 * torch.real(Ey * torch.conj(Hz) - Ez * torch.conj(Hy))
    Sy = 0.5 * torch.real(Ez * torch.conj(Hx) - Ex * torch.conj(Hz))
    Sz = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
    Snorm = torch.sqrt(Sx**2 + Sy**2 + Sz**2)

    # Calculate 2D boundary if it's an xz or yz cross section, or contours for xy
    if slice_plane == 'xz':
        if params_y is not None:
            y_eval = torch.tensor([slice_val], device=dev, dtype=geo_dtype)
            z_wavy, grating_height = get_continuous_boundary(x_plot, params_x, grating_period, y=y_eval, params_y=params_y, grating_period_y=grating_period_y)
            z_wavy_np = z_wavy.squeeze(1).cpu().numpy()
        else:
            z_wavy, grating_height = get_continuous_boundary(x_plot, params_x, grating_period)
            z_wavy_np = z_wavy.cpu().numpy()
        z_top = grating_height + h
        
    elif slice_plane in ['yz', 'zy']:
        if params_y is not None:
            x_eval = torch.tensor([slice_val], device=dev, dtype=geo_dtype)
            z_wavy, grating_height = get_continuous_boundary(x=x_eval, params_x=params_x, grating_period=grating_period, y=y_plot, params_y=params_y, grating_period_y=grating_period_y)
            z_wavy_np = z_wavy.squeeze(0).cpu().numpy()
            z_top = grating_height + h
        else:
            # If there's no Y grating, the profile is constant along Y
            z_wavy, grating_height = get_continuous_boundary(torch.tensor([slice_val], device=dev, dtype=geo_dtype), params_x, grating_period)
            z_wavy_np = np.full_like(v1_cpu, z_wavy.item())
            z_top = grating_height + h
            
    elif slice_plane == 'xy':
        if params_y is not None:
            z_wavy, grating_height = get_continuous_boundary(x_plot, params_x, grating_period, y=y_plot, params_y=params_y, grating_period_y=grating_period_y)
        else:
            z_wavy, grating_height = get_continuous_boundary(x_plot, params_x, grating_period)
            z_wavy = z_wavy.unsqueeze(1).expand(-1, len(y_plot))
        z_wavy_np = z_wavy.cpu().numpy()

    def format_ax(ax_to_format):
        if slice_plane in ['xz', 'yz', 'zy']:
            ax_to_format.plot(v1_cpu, z_wavy_np, color='black', linewidth=thickness+2 if thickness>0 else 0, linestyle='-')
            ax_to_format.plot([v1_cpu[0], v1_cpu[-1]], [z_top, z_top], color='black', linewidth=thickness+2, linestyle='-')
            ax_to_format.plot(v1_cpu, z_wavy_np, color='white', linewidth=thickness, linestyle='-')
            ax_to_format.plot([v1_cpu[0], v1_cpu[-1]], [z_top, z_top], color='white', linewidth=thickness, linestyle='-')
        elif slice_plane == 'xy':
            if 0 <= slice_val <= grating_height:
                try:
                    ax_to_format.contour(v1_cpu, v2_cpu, z_wavy_np.T, levels=[slice_val], colors='white', linewidths=thickness, alpha=0.8)
                except ValueError:
                    pass # Fails if slice_val is exactly out of bounds of the surface
        ax_to_format.set_ylim([v2_cpu[0], v2_cpu[-1]])
        ax_to_format.set_xlim([v1_cpu[0], v1_cpu[-1]])

    if field == 'sine_eps':
        if params_y is not None:
            _, grating_height_val = get_continuous_boundary(x_plot, params_x, grating_period, y=y_plot, params_y=params_y, grating_period_y=grating_period_y)
        else:
            _, grating_height_val = get_continuous_boundary(x_plot, params_x, grating_period)
        grating_height_val = float(grating_height_val)
        
        effective_n_layers = config.n_layers
        if config.height_per_layer is not None:
            effective_n_layers = max(1, math.ceil(grating_height_val / config.height_per_layer))

        eps_high = get_material_eps(config.grating_material, wavelength)
        sine_eps = get_staircase_sine_eps(x_plot, params_x, grating_period, effective_n_layers, eps_high, subpixel=config.subpixel, y=y_plot, params_y=params_y, grating_period_y=grating_period_y)
        
        fig, ax = plt.subplots(figsize=(8, 4))

        if slice_plane == 'xz':
            h_min, h_max = x_plot.min().item(), x_plot.max().item()
            v_min, v_max = 0.0, grating_height_val
            xlabel, ylabel = 'x (nm)', 'z (nm)'
            if params_y is not None:
                if y_plot is None: raise ValueError("y_plot required for xz slice of 3D profile")
                y_idx = torch.argmin(torch.abs(y_plot - slice_val)).item()
                plot_eps = sine_eps[:, y_idx, :]
            else:
                plot_eps = sine_eps
        elif slice_plane in ['yz', 'zy']:
            if params_y is None:
                raise ValueError("yz slice requires 3D params_y")
            if y_plot is None: raise ValueError("y_plot required for yz slice")
            h_min, h_max = y_plot.min().item(), y_plot.max().item()
            v_min, v_max = 0.0, grating_height_val
            xlabel, ylabel = 'y (nm)', 'z (nm)'
            x_idx = torch.argmin(torch.abs(x_plot - slice_val)).item()
            plot_eps = sine_eps[x_idx, :, :]
        elif slice_plane == 'xy':
            h_min, h_max = x_plot.min().item(), x_plot.max().item()
            if params_y is not None and y_plot is not None:
                v_min, v_max = y_plot.min().item(), y_plot.max().item()
            else:
                v_min, v_max = -grating_period/2, grating_period/2
            xlabel, ylabel = 'x (nm)', 'y (nm)'
            cell_height = grating_height_val / effective_n_layers
            z_idx = int(slice_val / cell_height)
            z_idx = min(max(z_idx, 0), effective_n_layers - 1)
            if params_y is not None:
                plot_eps = sine_eps[:, :, z_idx]
            else:
                plot_eps = sine_eps[:, z_idx].unsqueeze(1).expand(-1, 2)

        im = ax.imshow(
            plot_eps.cpu().abs().T,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='none',
            extent=[h_min, h_max, v_min, v_max],
        )

        ax.set_xticks(np.linspace(h_min, h_max, 6))
        ax.set_yticks(np.linspace(v_min, v_max, 6))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Staircase Permittivity Profile ({slice_plane} at {slice_val:.1f}nm)\n{title_base}')
        fig.colorbar(im, ax=ax, label='permittivity')
        fig.tight_layout()
        return fig, ax

    if field is not None:
        # Dictionary for single-plot selection
        field_dict = {
            'E norm': Enorm, 'Ex': torch.real(Ex).abs(), 'Ey': torch.real(Ey).abs(), 'Ez': torch.real(Ez).abs(),
            'H norm': Hnorm, 'Hx': torch.real(Hx).abs(), 'Hy': torch.real(Hy).abs(), 'Hz': torch.real(Hz).abs(),
            'S norm': Snorm, 'Sx': Sx.abs(), 'Sy': Sy.abs(), 'Sz': Sz.abs()
        }
        if field not in field_dict:
            raise ValueError(f"Field '{field}' not recognized. Choose from: {list(field_dict.keys())}")

        fig, ax = plt.subplots(figsize=(5, 8))
        plot_tensor = field_dict[field]
        im = ax.imshow(plot_tensor.T.cpu(), cmap='jet', origin='lower', extent=extent)
        format_ax(ax)
        title_str = (f'{field} (real abs)' if 'norm' not in field else field) + f"\n{title_base}"
        ax.set(title=title_str, xlabel=xlabel, ylabel=ylabel)
        
        cbar_label = 'S (A.U.)' if 'S' in field else ('H (A.U.)' if 'H' in field else 'E (A.U.)')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(cbar_label)
        fig.tight_layout()
        return fig, ax

    # 3x4 Grid Plotting
    fig, axes = plt.subplots(figsize=(15, 12), nrows=3, ncols=4)
    row0_imgs = [Enorm, torch.real(Ex).abs(), torch.real(Ey).abs(), torch.real(Ez).abs()]
    row1_imgs = [Hnorm, torch.real(Hx).abs(), torch.real(Hy).abs(), torch.real(Hz).abs()]
    row2_imgs = [Snorm, Sx.abs(), Sy.abs(), Sz.abs()]

    row0_vmin = min([x.min().item() for x in row0_imgs])
    row0_vmax = max([x.max().item() for x in row0_imgs])
    row1_vmin = min([x.min().item() for x in row1_imgs])
    row1_vmax = max([x.max().item() for x in row1_imgs])
    row2_vmin = min([x.min().item() for x in row2_imgs])
    row2_vmax = max([x.max().item() for x in row2_imgs])

    titles = [
        ['E norm', 'Ex real abs', 'Ey real abs', 'Ez real abs'],
        ['H norm', 'Hx real abs', 'Hy real abs', 'Hz real abs'],
        ['S norm', 'Sx abs', 'Sy abs', 'Sz abs']
    ]
    imgs = [row0_imgs, row1_imgs, row2_imgs]
    vmins = [row0_vmin, row1_vmin, row2_vmin]
    vmaxs = [row0_vmax, row1_vmax, row2_vmax]
    
    cbars_ims = []
    for r in range(3):
        for c in range(4):
            im = axes[r,c].imshow(imgs[r][c].T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=vmins[r], vmax=vmaxs[r])
            axes[r,c].set(title=titles[r][c], xlabel=xlabel, ylabel=ylabel)
            format_ax(axes[r,c])
            if c == 0: cbars_ims.append(im)
    if slice_plane == 'xy':
        fig.subplots_adjust(right=0.92, hspace=0.45, wspace=0.45)
    else:
        fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.30)
        
    cbar0 = fig.colorbar(cbars_ims[0], ax=axes[0, :], location='right', shrink=0.9, pad=0.02)
    cbar0.set_label('E (A.U.)')
    cbar1 = fig.colorbar(cbars_ims[1], ax=axes[1, :], location='right', shrink=0.9, pad=0.02)
    cbar1.set_label('H (A.U.)')
    cbar2 = fig.colorbar(cbars_ims[2], ax=axes[2, :], location='right', shrink=0.9, pad=0.02)
    cbar2.set_label('S (A.U.)')
    
    
    fig.suptitle(title_base, fontsize=16)
    
    return fig, axes

def generate_test_batch(stats, n_samples_per_mat=100):
    
    materials = stats["materials"] if isinstance(stats["materials"], list) else list(stats["materials"].keys())
    target_key = stats["target_key"]
    
    test_dir = Path(__file__).resolve().parent.parent / "Data" / "Test_Data"
    
    all_geo, all_px, all_mat, all_target = [], [], [], []
    for mat_id, mat_name in enumerate(materials):
        batch_path = test_dir / f"{mat_name}.pt"
        if not batch_path.exists():
            print(f"Warning: Test batch not found: {batch_path}")
            continue
            
        data = torch.load(batch_path, map_location="cpu", weights_only=False)
        
        def process_target(key, override_inc_ang=None):
            target = data[key][:n_samples_per_mat].float()
            if target.dim() == 2 and target.shape[1] == 2:
                target = torch.cat([target[:, 0], target[:, 1]], dim=-1)
            elif target.dim() == 3:
                target = torch.cat([target[:, :, 0], target[:, :, 1]], dim=-1)
                
            valid_mask = (target.max(dim=-1).values <= 1.0) & (target.min(dim=-1).values >= 0.0)
            
            if valid_mask.any():
                px = data["params_x"][:n_samples_per_mat].float()[valid_mask]
                all_px.append(px)
                
                geo_parts = [px.view(px.shape[0], -1)]
                geo_parts.append(data["h"][:n_samples_per_mat].float()[valid_mask].unsqueeze(-1))
                if override_inc_ang is not None:
                    geo_parts.append(torch.full((valid_mask.sum().item(), 1), override_inc_ang, dtype=torch.float32))
                else:
                    geo_parts.append(data["inc_ang"][:n_samples_per_mat].float()[valid_mask].unsqueeze(-1))
                all_geo.append(torch.cat(geo_parts, dim=-1))
                
                all_mat.append(torch.full((valid_mask.sum().item(),), mat_id, dtype=torch.long))
                all_target.append(target[valid_mask])
                
        if target_key == "all_film":
            process_target("A_film_normal", override_inc_ang=0.0)
            process_target("A_film_oblique", override_inc_ang=None)
        else:
            process_target(target_key, override_inc_ang=None)
        
    if not all_geo:
        raise FileNotFoundError(f"No test batches found in {test_dir}. Did you run the generation script and move them?")
        
    return {
        "geometry": torch.cat(all_geo),
        "params_x": torch.cat(all_px),
        "material_id": torch.cat(all_mat),
        "target": torch.cat(all_target).float()
    }
