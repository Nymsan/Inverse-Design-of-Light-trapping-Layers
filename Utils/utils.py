import numpy as np
import torch
from matplotlib import pyplot as plt
import torcwa
from pvlib import spectrum
from refractiveindex import RefractiveIndexMaterial
from tqdm import tqdm

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_staircase_sine_eps(x, params, grating_period, num_layers, eps_high, eps_low=1.,subpixel=True):
    """Generate sine grating in staircase approximation.

    Args:
        x (torch.tensor): 1D tensor of x positions.
        params (torch.Tensor): list of amplitude and phase shift. shape (n,2).
        grating_period (float): Period of the grating.
        num_layers (int): number of layers for stepwise approximation.
        eps_high (complex): Permittivity of high-index material.
        eps_low (complex, optional): Permittivity of low-index material. Defaults to 1.

    Returns:
        torch.tensor: 2D tensor of permittivity profile with shape (nx,num_layers).
    """
    grating_height = 2*torch.sum(params[:, 0]) + 1e-9
    freqs = torch.arange(1, params.shape[0]+1, dtype=x.dtype, device=x.device).unsqueeze(1)
    cosines = torch.cos(2. * np.pi * freqs * (x.unsqueeze(0) / grating_period) - params[:, 1].unsqueeze(1))
    cosines = cosines * params[:, 0].unsqueeze(1)
    eps_profile = grating_height/2 + torch.sum(cosines, dim=0)

    nx = x.shape[0]
    cell_height = grating_height / num_layers
    
    thresholds = torch.arange(0, num_layers, device=x.device) * cell_height
    thresholds = thresholds.unsqueeze(0).expand(nx,-1)
    eps_profile = eps_profile.unsqueeze(1).expand(-1,num_layers)

    eps = torch.clamp((eps_profile - thresholds) / cell_height,min=0,max=1)
    if not subpixel:
        eps = torch.round(eps)
    return eps_low + (eps_high-eps_low)*eps

def combine_staircases_to_3D(staircase_x,staircase_y):
    nx, ny = staircase_x.shape[0], staircase_y.shape[0]
    staircase_x = staircase_x.unsqueeze(1).expand(-1,ny,-1)
    staircase_y = staircase_y.unsqueeze(0).expand(nx,-1,-1)
    return stai

# light
spectra = spectrum.get_reference_spectra()
am15g = spectra['global']
def sun_weights(w):
    return torch.tensor(am15g[w.cpu().numpy()], dtype=geo_dtype, device=device)

# material
si = RefractiveIndexMaterial(shelf='main', book='Si', page='Green-2008')
def si_eps(w):
    return torch.tensor(si.get_refractive_index(w) + 1j * si.get_extinction_coefficient(w), 
                        dtype=sim_dtype, device=device)**2

ag = RefractiveIndexMaterial(shelf='main', book='Ag', page='McPeak')
def ag_eps(w):
    return torch.tensor(ag.get_refractive_index(w) + 1j * ag.get_extinction_coefficient(w), 
                        dtype=sim_dtype, device=device)**2

def get_continuous_boundary(x, params, grating_period):
    """Extracts the continuous 1D wavy boundary, corrected for peak-to-peak height."""
    grating_height = 2*torch.sum(params[:, 0]) + 1e-9
    freqs = torch.arange(1, params.shape[0]+1, dtype=x.dtype, device=x.device).unsqueeze(1)
    cosines = torch.cos(2. * np.pi * freqs * (x.unsqueeze(0) / grating_period) - params[:, 1].unsqueeze(1))
    cosines = cosines * params[:, 0].unsqueeze(1)
    
    # Original profile
    original_profile = grating_height/2 + torch.sum(cosines, dim=0)
    
    # Invert it because torcwa layers were added in reverse (-1-i)
    z_boundary = grating_height - original_profile

    return z_boundary, grating_height.item()
    
def get_incident_power(pol,wavelength,inc_ang,azi_ang,grating_period,order):
    L = [grating_period,grating_period]
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device, avoid_Pinv_instability=True)
    sim.add_input_layer()
    sim.add_output_layer()
    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=pol, direction='forward', notation='ps')
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torch.tensor([0],device=device), torch.tensor([0]), y=0.0)
    P_inc = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))

    return P_inc.item()*grating_period

def get_absorptance(params, wavelength=torch.tensor(700, dtype=geo_dtype), inc_ang=0, azi_ang=0, grating_period=1000, h=1000,
                    order_N=40, nx=5000, ny=1, n_layers=100, add_reflector=False, reflector_type='pec',subpixel=True):
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = grating_period
    torcwa.rcwa_geo.Ly = grating_period
    torcwa.rcwa_geo.nx = nx
    torcwa.rcwa_geo.ny = ny
    torcwa.rcwa_geo.grid()
    L = [grating_period,grating_period]
    
    order = [order_N, 0]

    grating_height = 2*torch.sum(params[:, 0])
    sine_eps = get_staircase_sine_eps(x=torcwa.rcwa_geo.x, params=params, grating_period=grating_period,
                                         num_layers=n_layers, eps_high=si_eps(wavelength),subpixel=subpixel if n_layers>1 else True)
        
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device, avoid_Pinv_instability=True)
    sim.add_input_layer()
    if add_reflector:
        if reflector_type == 'pec':
            reflector_eps = torch.tensor(-10000.0 + 0.0j, dtype=sim_dtype, device=device)

        elif reflector_type == 'Ag':
            reflector_eps = ag_eps(wavelength)

        sim.add_output_layer(eps=reflector_eps)
    else:
        sim.add_output_layer

    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    
    for i in range(sine_eps.shape[1]):
        sim.add_layer(thickness=grating_height/sine_eps.shape[1], eps=sine_eps[:, -1-i, None])
        
    sim.add_layer(thickness=h, eps=si_eps(wavelength))
    

    sim.solve_global_smatrix()

    z_top = grating_height # top of bulk layer, bottom of grating
    z_bot = grating_height + h #bottom of bulk
    z_air = torch.tensor(-5 * h, device=device, dtype=geo_dtype)
    
    results = {}
    for pol_idx, pol in enumerate([[1., 0.], [0., 1.]]): #first result is p-pol, second s-pol
        P_inc = get_incident_power(pol=pol,wavelength=wavelength,inc_ang=inc_ang,azi_ang=azi_ang,grating_period=grating_period,order=order)
        sim.source_planewave(amplitude=pol, direction='forward', notation='ps')
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torcwa.rcwa_geo.x, torch.stack((z_top, z_bot, z_air)), y=0.0)
        S_z = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
        
        P_top = torch.trapezoid(S_z[:, 0], torcwa.rcwa_geo.x)
        P_bot = torch.trapezoid(S_z[:, 1], torcwa.rcwa_geo.x)
        P_air = torch.trapezoid(S_z[:, 2], torcwa.rcwa_geo.x)
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

def get_weighted_absorptance(params, wavelengths,
                             inc_ang=0, azi_ang=0, grating_period=1000, h=1000, order_N=40, nx=5000, ny=1,  n_layers=100, 
                             add_reflector=False, reflector_type='pec',subpixel=True):
    L = [float(grating_period), 1.0]
    weights = sun_weights(wavelengths)
    sum_am15g = torch.sum(weights)
    sum_photons = torch.sum(weights * wavelengths.to(device))
    
    running_sun_weight = 0.0
    running_photon_weight = 0.0
    
    for i, wavelength in enumerate(wavelengths):
        A_film = get_absorptance(params=params, wavelength=wavelength, inc_ang=inc_ang, azi_ang=azi_ang,
                                                  grating_period=grating_period, n_layers=n_layers, h=h, 
                                                  order_N=order_N, nx=nx, add_reflector=add_reflector,
                                                  reflector_type=reflector_type,subpixel=subpixel)[2]
        mean_A = torch.mean(A_film)
        running_sun_weight += weights[i] * mean_A
        running_photon_weight += weights[i] * mean_A * wavelength.to(device)
        
    weighted_A_sun = running_sun_weight / sum_am15g
    weighted_A_photon = running_photon_weight / sum_photons
    return weighted_A_sun, weighted_A_photon

def get_absorptance_curve(params, wavelengths,
                             inc_ang=0, azi_ang=0, grating_period=1000, h=1000, order_N=40, nx=5000, ny=1,  n_layers=100, 
                             add_reflector=False, reflector_type='pec',subpixel=True):
    Absorptances_film = torch.zeros_like(wavelengths,dtype=geo_dtype).unsqueeze(1).repeat(1,2)
    Absorptances_grating = torch.zeros_like(wavelengths,dtype=geo_dtype).unsqueeze(1).repeat(1,2)
    for i,wavelength in enumerate(tqdm(wavelengths)):
        A_film,A_grating = get_absorptance(params=params, wavelength=wavelength, inc_ang=inc_ang, azi_ang=azi_ang,
                                                  grating_period=grating_period, n_layers=n_layers, h=h, 
                                                  order_N=order_N, nx=nx, add_reflector=add_reflector,
                                                  reflector_type=reflector_type,subpixel=subpixel)[2:4]

        Absorptances_film[i,:] = A_film.cpu()
        Absorptances_grating[i,:] = A_grating.cpu()
    return Absorptances_film, Absorptances_grating

    

def plot_fields(sim, x_plot, z_plot, wavelength, polarization, inc_ang, azi_ang, params, grating_period, h, field=None):
    """
    Plots fields. If `field` is None, plots a 4x3 grid.
    If `field` is a string (e.g., 'Ex', 'Snorm'), plots only that field.
    """
    sim.source_planewave(amplitude=polarization, direction='forward', notation='ps')
    
    # Ensure inputs are on the same device as the simulation
    dev = sim._device
    x_plot = x_plot.to(dev)
    z_plot = z_plot.to(dev)
    
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(x_plot, z_plot, torch.tensor(0.0, device=dev))
    
    # Field magnitudes
    Enorm = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
    Hnorm = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)
    
    # Poynting vector components
    Sx = 0.5 * torch.real(Ey * torch.conj(Hz) - Ez * torch.conj(Hy))
    Sy = 0.5 * torch.real(Ez * torch.conj(Hx) - Ex * torch.conj(Hz))
    Sz = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
    Snorm = torch.sqrt(Sx**2 + Sy**2 + Sz**2)

    x_cpu = x_plot.cpu().numpy()
    z_cpu = z_plot.cpu().numpy()
    extent = [x_cpu[0], x_cpu[-1], z_cpu[0], z_cpu[-1]]

    # Calculate the boundaries once
    z_wavy, grating_height = get_continuous_boundary(x_plot, params, grating_period)
    z_wavy_np = z_wavy.cpu().numpy()
    z_top = grating_height + h
    
    title_base = f"xz-plane field distribution at $\\lambda$ = {wavelength} nm and y = 0 nm \n polarization -> {polarization} in ps basis\n incident angle = {inc_ang*180/np.pi:.1f}°, azimuthal angle = {azi_ang*180/np.pi:.1f}°"

    if field is not None:
        # Dictionary for single-plot selection
        field_dict = {
            'Enorm': Enorm, 'Ex': torch.real(Ex).abs(), 'Ey': torch.real(Ey).abs(), 'Ez': torch.real(Ez).abs(),
            'Hnorm': Hnorm, 'Hx': torch.real(Hx).abs(), 'Hy': torch.real(Hy).abs(), 'Hz': torch.real(Hz).abs(),
            'Snorm': Snorm, 'Sx': Sx.abs(), 'Sy': Sy.abs(), 'Sz': Sz.abs()
        }
        
        if field not in field_dict:
            raise ValueError(f"Field '{field}' not recognized. Choose from: {list(field_dict.keys())}")

        fig, ax = plt.subplots(figsize=(5, 8))
        plot_tensor = field_dict[field]
        
        im = ax.imshow(plot_tensor.T.cpu(), cmap='jet', origin='lower', extent=extent)
        
        # Geometry outline
        ax.plot(x_cpu, z_wavy_np, color='black', linewidth=5, linestyle='-')
        ax.plot([x_cpu[0], x_cpu[-1]], [z_top, z_top], color='black', linewidth=5, linestyle='-')
        ax.plot(x_cpu, z_wavy_np, color='white', linewidth=3, linestyle='-')
        ax.plot([x_cpu[0], x_cpu[-1]], [z_top, z_top], color='white', linewidth=3, linestyle='-')
        ax.set_ylim([z_cpu[0], z_cpu[-1]])
        ax.set_xlim([x_cpu[0], x_cpu[-1]])
        
        # Formatting
        ax.set(title=f'{field} (real abs)' if 'norm' not in field else field, xlabel='x (nm)', ylabel='z (nm)')
        
        # Dynamic colorbar label
        cbar_label = 'S (A.U.)' if 'S' in field else ('H (A.U.)' if 'H' in field else 'E (A.U.)')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(cbar_label)
        
        #fig.suptitle(title_base, fontsize=12)
        fig.tight_layout()
        
        return fig, ax

    # --- 3x4 Grid Plotting Logic ---
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

    # Row 0: E fields
    im0 = axes[0,0].imshow(Enorm.T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,0].set(title='E norm', xlabel='x (nm)', ylabel='z (nm)')
    axes[0,1].imshow(torch.real(Ex).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,1].set(title='Ex real abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[0,2].imshow(torch.real(Ey).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,2].set(title='Ey real abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[0,3].imshow(torch.real(Ez).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,3].set(title='Ez real abs', xlabel='x (nm)', ylabel='z (nm)')

    # Row 1: H fields
    im4 = axes[1,0].imshow(Hnorm.T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,0].set(title='H norm', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,1].imshow(torch.real(Hx).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,1].set(title='Hx real abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,2].imshow(torch.real(Hy).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,2].set(title='Hy real abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,3].imshow(torch.real(Hz).abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,3].set(title='Hz real abs', xlabel='x (nm)', ylabel='z (nm)')

    # Row 2: Poynting
    im8 = axes[2,0].imshow(Snorm.T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,0].set(title='S norm', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,1].imshow(Sx.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,1].set(title='Sx abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,2].imshow(Sy.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,2].set(title='Sy abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,3].imshow(Sz.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,3].set(title='Sz abs', xlabel='x (nm)', ylabel='z (nm)')
    
    for ax in axes.flat:
        ax.plot(x_cpu, z_wavy_np, color='white', linewidth=1, linestyle='-')
        ax.plot([x_cpu[0], x_cpu[-1]], [z_top, z_top], color='white', linewidth=1, linestyle='-')
        ax.set_ylim([z_cpu[0], z_cpu[-1]])
        ax.set_xlim([x_cpu[0], x_cpu[-1]])

    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.3)

    cbar0 = fig.colorbar(im0, ax=axes[0, :], location='right', shrink=0.9, pad=0.02)
    cbar0.set_label('E (A.U.)')
    cbar1 = fig.colorbar(im4, ax=axes[1, :], location='right', shrink=0.9, pad=0.02)
    cbar1.set_label('H (A.U.)')
    cbar2 = fig.colorbar(im8, ax=axes[2, :], location='right', shrink=0.9, pad=0.02)
    cbar2.set_label('S (A.U.)')
    fig.suptitle(title_base, fontsize=16)
    
    return fig, axes
