import numpy as np
import torch
from matplotlib import pyplot as plt
import torcwa
from tqdm.notebook import tqdm
from pvlib import spectrum
from refractiveindex import RefractiveIndexMaterial
import os

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cuda')

def get_sine_eps(x,params,grating_period,eps):
    """Generate sine grating permittivity profile.

    Args:
        x (torch.tensor): 1D tensor of x positions.
        params (torch.Tensor): list of amplitude and phase shift. shape (n,2), where n is n*2*np.pi/grating_period'th frequency.
        eps (float): Permittivity of high-index material.

    Returns:
        torch.tensor: 1D tensor of permittivity profile.
    """
    A = torch.sum(params[:,0]) + 1e-9
    cosines = torch.cos(2.*np.pi*torch.arange(1, params.shape[0]+1, 
                                              dtype=geo_dtype,device=device).unsqueeze(1)*(x.unsqueeze(0)/grating_period)
                                               - params[:,1].unsqueeze(1))
    cosines = cosines * params[:,0].unsqueeze(1)
    eps = 1 + (eps-1)*(0.5*(A+torch.sum(cosines, dim=0))/A)
    return eps.unsqueeze(1)   # make shape (nx,1) so add_layer accepts it

# light
spectra = spectrum.get_reference_spectra()
am15g = spectra['global']
sun_weights = lambda x: torch.tensor(am15g[x.cpu().numpy()])

# material
si = RefractiveIndexMaterial(shelf='main', book='Si', page='Green-2008')
si_eps = lambda x: torch.tensor(si.get_refractive_index(x) +
                      1j * si.get_extinction_coefficient(x))**2

def get_absorptance(params,wavelength,inc_ang,azi_ang,grating_period,h,order_N,L,nx):
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = L[0]
    torcwa.rcwa_geo.Ly = L[1]
    torcwa.rcwa_geo.ny = 1
    order = [order_N,0]
    # calculate incidence power from inc_ang, azi_ang
    P_inc = 0.5 / np.cos(inc_ang) * (np.cos(azi_ang)**2 + np.sin(azi_ang)**2*np.cos(inc_ang)**2)
    #See Silicon Modulation Sanity Checks.ipynb for more details
    P_inc = P_inc*torcwa.rcwa_geo.Lx

    #setup simulation
    torcwa.rcwa_geo.nx = nx #set sine grating grid
    torcwa.rcwa_geo.grid()
    A = torch.sum(params[:,0])
    sine_eps = get_sine_eps(torcwa.rcwa_geo.x,params=params,grating_period=grating_period,eps=si_eps(wavelength))
    sim = torcwa.rcwa(freq=1/wavelength,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer()
    sim.add_output_layer()
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    sim.add_layer(thickness=A,eps=sine_eps)
    sim.add_layer(thickness=h,eps=si_eps(wavelength))
    sim.solve_global_smatrix()

    # choose probe planes (just above / below film). use same device dtype as sim
    z_top = torch.clone(A)  # e.g. top of film
    z_bot = torch.clone(A+h)  # e.g. bottom of film 
    z_air = torch.tensor(-3*h,device=sim._device, dtype=geo_dtype) #Ensure evanescent modes are gone
    torcwa.rcwa_geo.nx = nx #set sampling grid for fields
    torcwa.rcwa_geo.grid()
    P_absorbed_film = torch.zeros(2,device=sim._device, dtype=geo_dtype) # to store absorbed power in film, first item is x polarization, second y polarization
    P_absorbed_grating = torch.zeros(2,device=sim._device, dtype=geo_dtype) # to store absorbed power in grating, first item is x polarization, second y polarization
    A_film = torch.zeros(2,device=sim._device, dtype=geo_dtype) #to store absorptance in film. first item is x polarization, second y polarization
    A_grating = torch.zeros(2,device=sim._device, dtype=geo_dtype) #to store absorptance in grating. first item is x polarization, second y polarization
    P_slices = torch.zeros((2,3),device=sim._device, dtype=geo_dtype) # to store Poynting vector slices at each plane for both polarizations
    Reflectance = torch.zeros(2,device=sim._device, dtype=geo_dtype) # to store reflectance for both polarizations first item is x polarization, second y polarization
    Transmittance = torch.zeros(2,device=sim._device, dtype=geo_dtype) # to store transmittance for both polarizations first item is x polarization, second y polarization

    sim.source_planewave(amplitude=[1.,0.],direction='forward',notation='xy')
    # request fields at both planes: x_axis is your x sampling (1D tensor), y0 is y coordinate (often 0)
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torcwa.rcwa_geo.x, torch.stack((z_top,z_bot,z_air)), y=0.0)
    # Ex,Hy shapes: (nx, 2)  (nx across x, 2 planes)
    S_z = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))   # shape (nx,3)
    P_top = torch.trapezoid(S_z[:,0], torcwa.rcwa_geo.x)
    P_bot = torch.trapezoid(S_z[:,1], torcwa.rcwa_geo.x)
    P_air = torch.trapezoid(S_z[:,2], torcwa.rcwa_geo.x)
    P_absorbed_film[0] = P_top - P_bot
    P_absorbed_grating[0] = P_air - P_top
    A_film[0] = P_absorbed_film[0] / P_inc
    A_grating[0] = P_absorbed_grating[0] / P_inc
    P_slices[0,:] = torch.tensor([P_top, P_bot, P_air], device=sim._device, dtype=geo_dtype)
    Reflectance[0]  = (P_inc - P_air) / P_inc
    Transmittance[0] = P_bot / P_inc

    sim.source_planewave(amplitude=[0.,1.],direction='forward',notation='xy')
    # request fields at both planes: x_axis is your x sampling (1D tensor), y0 is y coordinate (often 0)
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(torcwa.rcwa_geo.x, torch.stack((z_top,z_bot,z_air)), y=0.0)
    # Ex,Hy shapes: (nx, 2)  (nx across x, 2 planes)
    S_z = 0.5 * torch.real(Ex * torch.conj(Hy) - Ey * torch.conj(Hx))   # shape (nx,3)
    P_top = torch.trapezoid(S_z[:,0], torcwa.rcwa_geo.x)
    P_bot = torch.trapezoid(S_z[:,1], torcwa.rcwa_geo.x)
    P_air = torch.trapezoid(S_z[:,2], torcwa.rcwa_geo.x)
    P_absorbed_film[1] = P_top - P_bot
    P_absorbed_grating[1] = P_air - P_top
    A_film[1] = P_absorbed_film[1] / P_inc
    A_grating[1] = P_absorbed_grating[1] / P_inc
    P_slices[1,:] = torch.tensor([P_top, P_bot, P_air], device=sim._device, dtype=geo_dtype)
    Reflectance[1]  = (P_inc - P_air) / P_inc
    Transmittance[1] = P_bot / P_inc

    return  A_film, A_grating, Reflectance, Transmittance, P_absorbed_film, P_absorbed_grating, P_slices

def get_weighted_absorptance(params,wavelengths=torch.linspace(300,1100,100,dtype=int),
                             inc_ang=0,azi_ang=0,grating_period=1000,h=1000,order_N=40, nx=5000):
    L = [grating_period,1.] #nm
    sum_am15g = torch.sum(torch.tensor([sun_weights(wavelength) for wavelength in wavelengths], device=device))
    sum_photons = torch.sum(torch.tensor([sun_weights(wavelength)*wavelength for wavelength in wavelengths], device=device))
    running_sun_weight = 0.0
    running_photon_weight = 0.0
    for wavelength in wavelengths:
        A_film, _, _, _, _, _, _ = get_absorptance(params,wavelength,inc_ang,azi_ang,grating_period,h,order_N,L,nx)
        running_sun_weight += sun_weights(wavelength) * torch.mean(A_film)
        running_photon_weight += sun_weights(wavelength) * torch.mean(A_film) * wavelength
    weighted_A_sun = running_sun_weight / sum_am15g
    weighted_A_photon = running_photon_weight / sum_photons
    return weighted_A_sun, weighted_A_photon
