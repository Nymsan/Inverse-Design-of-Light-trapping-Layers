import numpy as np
import torch
from matplotlib import pyplot as plt
import torcwa
from pvlib import spectrum
from refractiveindex import RefractiveIndexMaterial
from tqdm import tqdm
from torchaudio.transforms import Convolve

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_sine_eps(x, params, grating_period, eps,):
    """Generate sine grating permittivity profile.

    Args:
        x (torch.tensor): 1D tensor of x positions.
        params (torch.Tensor): list of amplitude and phase shift. shape (n,2), where n is n*2*np.pi/grating_period'th frequency.
        grating_period (float): Period of the grating.
        eps (complex): Permittivity of high-index material.

    Returns:
        torch.tensor: 1D tensor of permittivity profile.
    """
    A = torch.sum(params[:, 0]) + 1e-9
    freqs = torch.arange(0, params.shape[0], dtype=x.dtype, device=x.device).unsqueeze(1)
    cosines = torch.cos(2. * np.pi * freqs * (x.unsqueeze(0) / grating_period) - params[:, 1].unsqueeze(1))
    cosines = cosines * params[:, 0].unsqueeze(1)
    eps_profile = 1 + (eps - 1) * (0.5 * (A + torch.sum(cosines, dim=0)) / A)
    return eps_profile.unsqueeze(1)   # make shape (nx,1) so add_layer accepts it


def get_staircase_sine_eps(x, params, grating_period, num_layers, eps_high, eps_low=1.,Hann_window_length=0):
    """Generate sine grating in stepwise approximation.

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
    A = torch.sum(params[:, 0]) + 1e-9
    freqs = torch.arange(0, params.shape[0], dtype=x.dtype, device=x.device).unsqueeze(1)
    cosines = torch.cos(2. * np.pi * freqs * (x.unsqueeze(0) / grating_period) - params[:, 1].unsqueeze(1))
    cosines = cosines * params[:, 0].unsqueeze(1)
    eps_profile = 0.5 * (A + torch.sum(cosines, dim=0))
    
    eps = torch.full((x.shape[0], num_layers), eps_low, dtype=sim_dtype, device=x.device)
    thresholds = torch.arange(1, num_layers + 1, device=x.device) * (A / num_layers)
    
    for i in range(num_layers):
        eps[:, i] = torch.where(eps_profile > thresholds[i], 1.0, 0.0)
        if Hann_window_length > 0:
            hann = torch.hann_window(Hann_window_length)
            eps[:,i] = Convolve(mode='same')(hann,eps[:,i])
    return eps 

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

def get_absorptance(params, wavelength=torch.tensor(700, dtype=geo_dtype), inc_ang=0, azi_ang=0, grating_period=1000, h=1000,
                    order_N=40, nx=5000, ny=1, n_layers=100, add_reflector=False, reflector_type='pec'):
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = grating_period
    torcwa.rcwa_geo.Ly = grating_period
    torcwa.rcwa_geo.nx = nx
    torcwa.rcwa_geo.ny = ny
    torcwa.rcwa_geo.grid()
    L = [grating_period,grating_period]
    
    order = [order_N, 0]
    # calculate incidence power from inc_ang, azi_ang
    P_inc = 0.5 / np.cos(inc_ang) * (np.cos(azi_ang)**2 + np.sin(azi_ang)**2 * np.cos(inc_ang)**2)
    P_inc = P_inc * torcwa.rcwa_geo.Lx

    A = torch.sum(params[:, 0])
    if n_layers > 1:
        sine_eps = get_staircase_sine_eps(x=torcwa.rcwa_geo.x, params=params, grating_period=grating_period,
                                         num_layers=n_layers, eps_high=si_eps(wavelength))
    else:
        sine_eps = get_sine_eps(x=torcwa.rcwa_geo.x, params=params, grating_period=grating_period, eps=si_eps(wavelength))
        
    sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=sim_dtype, device=device)
    sim.add_input_layer()
    sim.add_output_layer()
    sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
    
    for i in range(sine_eps.shape[1]):
        sim.add_layer(thickness=A/sine_eps.shape[1], eps=sine_eps[:, -1-i, None])
        
    sim.add_layer(thickness=h, eps=si_eps(wavelength))
    
    if add_reflector:
        if reflector_type == 'pec':
            reflector_eps = torch.tensor(-10000.0 + 0.0j, dtype=sim_dtype, device=device)
        elif reflector_type == 'Ag':
            reflector_eps = ag_eps(wavelength)
        sim.add_layer(thickness=h/5, eps=reflector_eps)

    sim.solve_global_smatrix()

    z_top = A
    z_bot = A + h
    z_air = torch.tensor(-5 * h, device=device, dtype=geo_dtype)
    z_reflector = A+h+h/5
    
    results = {}
    for pol_idx, pol in enumerate([[1., 0.], [0., 1.]]):
        sim.source_planewave(amplitude=pol, direction='forward', notation='xy')
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torcwa.rcwa_geo.x, torch.stack((z_top, z_bot, z_air,z_reflector)), y=0.0)
        S_z = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))
        
        P_top = torch.trapezoid(S_z[:, 0], torcwa.rcwa_geo.x)
        P_bot = torch.trapezoid(S_z[:, 1], torcwa.rcwa_geo.x)
        P_air = torch.trapezoid(S_z[:, 2], torcwa.rcwa_geo.x)
        #Add a sanity check using a volume integral for the power.
        P_reflector = torch.trapezoid(S_z[:,3],torcwa.rcwa_geo.x)
        
        results[pol_idx] = {
            'A_film': (P_top - P_bot) / P_inc,
            'A_grating': (P_air - P_top) / P_inc,
            'A_reflector': (P_bot - P_reflector) / P_inc,
            'R': (P_inc - P_air) / P_inc,
            'T': P_bot / P_inc,
            'P_abs_film': P_top - P_bot,
            'P_abs_grating': P_air - P_top,
            'P_abs_reflector': P_top - P_reflector,
            'P_slices': torch.tensor([P_inc,P_top, P_bot, P_air, P_reflector], device=device)
        }

    A_film = torch.tensor([results[0]['A_film'], results[1]['A_film']], device=device)
    A_grating = torch.tensor([results[0]['A_grating'], results[1]['A_grating']], device=device)
    A_reflector = torch.tensor([results[0]['A_reflector'],results[1]['A_reflector']],device=device)
    Reflectance = torch.tensor([results[0]['R'], results[1]['R']], device=device)
    Transmittance = torch.tensor([results[0]['T'], results[1]['T']], device=device)
    P_abs_film = torch.tensor([results[0]['P_abs_film'], results[1]['P_abs_film']], device=device)
    P_abs_grating = torch.tensor([results[0]['P_abs_grating'], results[1]['P_abs_grating']], device=device)
    P_abs_reflector = torch.tensor([results[0]['P_abs_reflector'], results[1]['P_abs_reflector']], device=device)
    P_slices = torch.stack([results[0]['P_slices'], results[1]['P_slices']])

    return sim, sine_eps, A_film, A_grating, A_reflector, Reflectance, Transmittance, P_abs_film, P_abs_grating, P_abs_reflector, P_slices

def get_weighted_absorptance(params, wavelengths=torch.linspace(300, 1100, 100, dtype=int),
                             inc_ang=0, azi_ang=0, grating_period=1000, h=1000, order_N=40, nx=5000, ny=1,  n_layers=100, 
                             add_reflector=False, reflector_type='pec'):
    L = [float(grating_period), 1.0]
    weights = sun_weights(wavelengths)
    sum_am15g = torch.sum(weights)
    sum_photons = torch.sum(weights * wavelengths.to(device))
    
    running_sun_weight = 0.0
    running_photon_weight = 0.0
    
    for i, wavelength in enumerate(wavelengths):
        A_film = get_absorptance(params=params, wavelength=wavelength, inc_ang=inc_ang, azi_ang=azi_ang,
                                                  grating_period=grating_period, n_layers=n_layers, h=h, 
                                                  order_N=order_N, nx=nx, add_reflector=add_reflector, reflector_type=reflector_type)[2]
        mean_A = torch.mean(A_film)
        running_sun_weight += weights[i] * mean_A
        running_photon_weight += weights[i] * mean_A * wavelength.to(device)
        
    weighted_A_sun = running_sun_weight / sum_am15g
    weighted_A_photon = running_photon_weight / sum_photons
    return weighted_A_sun, weighted_A_photon

def get_absorptance_curve(params, wavelengths=torch.linspace(300, 1100, 100, dtype=int),
                             inc_ang=0, azi_ang=0, grating_period=1000, h=1000, order_N=40, nx=5000, ny=1,  n_layers=100, 
                             add_reflector=False, reflector_type='pec'):
    Absorptances = torch.zeros_like(wavelengths,dtype=torch.float32)
    for i,wavelength in enumerate(tqdm(wavelengths)):
        A_film = get_absorptance(params=params, wavelength=wavelength, inc_ang=inc_ang, azi_ang=azi_ang,
                                                  grating_period=grating_period, n_layers=n_layers, h=h, 
                                                  order_N=order_N, nx=nx, add_reflector=add_reflector, reflector_type=reflector_type)[2]
        Absorptances[i] = torch.mean(A_film).cpu()
    return Absorptances, wavelengths

    

def plot_fields(sim, x_plot, z_plot, wavelength, polarization, inc_ang, azi_ang, global_max=None):
    """
    Plots a 4x3 grid of fields:
    Row 0: |E|, |Ex|, |Ey|, |Ez|
    Row 1: |H|, |Hx|, |Hy|, |Hz|
    Row 2: |S|, Sx, Sy, Sz
    """
    sim.source_planewave(amplitude=polarization, direction='forward', notation='xy')
    
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

    fig, axes = plt.subplots(figsize=(15, 12), nrows=3, ncols=4)
    
    # compute per-row vmin/vmax so each row shares a common color scale
    row0_imgs = [Enorm, torch.abs(Ex), torch.abs(Ey), torch.abs(Ez)]
    row1_imgs = [Hnorm, torch.abs(Hx), torch.abs(Hy), torch.abs(Hz)]
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
    axes[0,1].imshow(torch.abs(Ex).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,1].set(title='Ex abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[0,2].imshow(torch.abs(Ey).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,2].set(title='Ey abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[0,3].imshow(torch.abs(Ez).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row0_vmin, vmax=row0_vmax)
    axes[0,3].set(title='Ez abs', xlabel='x (nm)', ylabel='z (nm)')

    # Row 1: H fields
    im4 = axes[1,0].imshow(Hnorm.T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,0].set(title='H norm', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,1].imshow(torch.abs(Hx).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,1].set(title='Hx abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,2].imshow(torch.abs(Hy).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,2].set(title='Hy abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[1,3].imshow(torch.abs(Hz).T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row1_vmin, vmax=row1_vmax)
    axes[1,3].set(title='Hz abs', xlabel='x (nm)', ylabel='z (nm)')

    # Row 2: Poynting
    im8 = axes[2,0].imshow(Snorm.T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,0].set(title='S norm', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,1].imshow(Sx.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,1].set(title='Sx abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,2].imshow(Sy.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,2].set(title='Sy abs', xlabel='x (nm)', ylabel='z (nm)')
    axes[2,3].imshow(Sz.abs().T.cpu(), cmap='jet', origin='lower', extent=extent, vmin=row2_vmin, vmax=row2_vmax)
    axes[2,3].set(title='Sz abs', xlabel='x (nm)', ylabel='z (nm)')

    # adjust layout to leave room on the right for three colorbars
    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.3)

    # one colorbar per row
    cbar0 = fig.colorbar(im0, ax=axes[0, :], location='right', shrink=0.9, pad=0.02)
    cbar0.set_label('E (A.U.)')
    cbar1 = fig.colorbar(im4, ax=axes[1, :], location='right', shrink=0.9, pad=0.02)
    cbar1.set_label('H (A.U.)')
    cbar2 = fig.colorbar(im8, ax=axes[2, :], location='right', shrink=0.9, pad=0.02)
    cbar2.set_label('S (A.U.)')
    title = f'xz-plane field distribution at $\\lambda$ = {wavelength} nm and y = 0 nm \n polarization = $E_x$ = {polarization[0]:.3f}, $E_y$ = {polarization[1]:.3f} \n incident angle = {inc_ang*180/np.pi:.1f}°, azimuthal angle = {azi_ang*180/np.pi:.1f}°'
    fig.suptitle(title, fontsize=16)
    
    return
