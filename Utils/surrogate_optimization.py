import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from tqdm import tqdm

class BatchedSurrogateOptimizer:
    def __init__(self, forward_model: nn.Module, geo_min: torch.Tensor, geo_max: torch.Tensor, nx: int = 128, device: str = "cuda", max_inc_deg: float = None, h_val: float = None, inc_val: float = None):
        self.forward_model = forward_model.to(device)
        self.forward_model.eval()
        
        self.geo_min = geo_min.clone().to(device)
        self.geo_max = geo_max.clone().to(device)
        
        if max_inc_deg is not None:
            self.geo_max[-1] = min(self.geo_max[-1].item(), max_inc_deg)
            
        if h_val is not None:
            self.geo_min[-2] = h_val
            self.geo_max[-2] = h_val
            
        if inc_val is not None:
            self.geo_min[-1] = inc_val
            self.geo_max[-1] = inc_val
            
        self.device = device
        self.nx = nx
        self.n_materials = 3
        
        self.n_fourier = len(self.geo_min) - 2
        
        self.h_min = self.geo_min[-2:-1]
        self.h_max = self.geo_max[-2:-1]
        self.inc_min = self.geo_min[-1:]
        self.inc_max = self.geo_max[-1:]
        for param in self.forward_model.parameters():
            param.requires_grad = False

    def _get_target_and_mask(self, bands: List[Tuple[float, float]], n_wavelengths: int = 322) -> Tuple[torch.Tensor, torch.Tensor]:
        wl_len = n_wavelengths // 2
        wls = torch.linspace(300, 1100, wl_len, device=self.device)
        
        target_p = torch.zeros(wl_len, device=self.device)
        mask_p = torch.zeros(wl_len, device=self.device)
        
        if not bands:
            target_p[:] = 1.0
            mask_p[:] = 1.0
        else:
            for b_min, b_max in bands:
                idx = (wls >= b_min) & (wls <= b_max)
                target_p[idx] = 1.0
                mask_p[idx] = 1.0
            
        # Duplicate for s-polarization
        target = torch.cat([target_p, target_p], dim=0).unsqueeze(0)
        mask = torch.cat([mask_p, mask_p], dim=0).unsqueeze(0)
        
        return target, mask

    def optimize_geometry(self, bands: List[Tuple[float, float]], n_restarts: int = 10000, steps: int = 300, lr: float = 0.1, allowed_materials: list[int] = None, top_k: int = 2, show_progress: bool = True) -> dict:
        self.forward_model.eval()
        target, mask = self._get_target_and_mask(bands)
        
        if allowed_materials is None:
            allowed_materials = list(range(self.n_materials))
            
        n_allowed = len(allowed_materials)
        B = n_restarts * n_allowed
        target = target.expand(B, -1)
        mask = mask.expand(B, -1)
        
        geo = torch.rand(B, len(self.geo_min), device=self.device) * (self.geo_max - self.geo_min) + self.geo_min
        geo = nn.Parameter(geo)
        
        mat = torch.tensor(allowed_materials, device=self.device).repeat_interleave(n_restarts)
        
        optimizer = optim.Adam([geo], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        history = torch.zeros((steps, n_allowed), device=self.device)
        pbar = tqdm(range(steps), desc="Optimizing Geometry", disable=not show_progress, leave=False)
        for step in pbar:
            optimizer.zero_grad()
            pred = self.forward_model(geometry=geo, material_id=mat)
            
            abs_vals = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
            
            # Boundary penalty: perfectly smooth and differentiable polynomial (N=10)
            # This only actively punishes the outer 1-5% of the bounds, allowing the model
            # to still reach the boundaries (like h=3000) if the physical performance gain outweighs it!
            range_geo = self.geo_max - self.geo_min
            active_mask = (range_geo > 1e-5).float()
            p_norm = (geo - self.geo_min) / (range_geo + 1e-9)
            boundary_penalty = 0.05 * torch.sum(active_mask * torch.pow((p_norm - 0.5) * 2.0, 10), dim=-1)
            
            loss = -abs_vals + boundary_penalty
            loss_sum = loss.mean()
            loss_sum.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                geo.clamp_(self.geo_min, self.geo_max)
                avg_abs = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
                
                range_geo = self.geo_max - self.geo_min
                active_mask = (range_geo > 1e-5).float()
                p_norm = (geo - self.geo_min) / (range_geo + 1e-9)
                boundary_penalty = 0.05 * torch.sum(active_mask * torch.pow((p_norm - 0.5) * 2.0, 10), dim=-1)
                penalized_abs = avg_abs - boundary_penalty
                
                penalized_reshaped = penalized_abs.view(n_allowed, n_restarts)
                max_indices = penalized_reshaped.argmax(dim=1)
                
                abs_reshaped = avg_abs.view(n_allowed, n_restarts)
                # Track the pure unpenalized absorptance of the winning penalized structures
                best_pure_abs = abs_reshaped[torch.arange(n_allowed), max_indices]
                history[step] = best_pure_abs
                
                from Utils.models import MATERIAL_LIBRARY
                postfix_dict = {
                    list(MATERIAL_LIBRARY.keys())[mat_idx]: f"{best_pure_abs[i].item():.4f}" 
                    for i, mat_idx in enumerate(allowed_materials)
                }
                pbar.set_postfix(postfix_dict)
                
        with torch.no_grad():
            pred = self.forward_model(geometry=geo, material_id=mat)
            final_abs = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
            range_geo = self.geo_max - self.geo_min
            active_mask = (range_geo > 1e-5).float()
            p_norm = (geo - self.geo_min) / (range_geo + 1e-9)
            boundary_penalty = 0.05 * torch.sum(active_mask * torch.pow((p_norm - 0.5) * 2.0, 10), dim=-1)
            final_abs = final_abs - boundary_penalty
            
        final_abs_reshaped = final_abs.view(n_allowed, n_restarts)
        top_vals, top_indices_local = final_abs_reshaped.topk(top_k, dim=1, largest=True)
        
        top_results = []
        for i in range(n_allowed):
            mat_idx = allowed_materials[i]
            for k in range(top_k):
                local_idx = top_indices_local[i, k].item()
                global_idx = i * n_restarts + local_idx
                top_results.append({
                    "material_idx": mat_idx,
                    "geometry": geo[global_idx].detach().cpu(),
                    "curve": pred[global_idx].detach().float().cpu(),
                    "loss": final_abs[global_idx].item()
                })
                
        best_global_idx = final_abs.argmax().item()
        
        return {
            "best_loss": final_abs[best_global_idx].item(),
            "best_geometry": geo[best_global_idx].detach().cpu(),
            "best_material": mat[best_global_idx].item(),
            "best_curve": pred[best_global_idx].detach().float().cpu(),
            "target": target[0].float().cpu(),
            "mask": mask[0].float().cpu(),
            "mode": "geometry",
            "history": history.cpu(),
            "top_results": top_results,
            "allowed_materials": allowed_materials
        }

    def optimize_profile(self, bands: List[Tuple[float, float]], n_restarts: int = 10000, steps: int = 300, lr: float = 5.0, allowed_materials: list[int] = None, top_k: int = 2, show_progress: bool = True) -> dict:
        self.forward_model.eval()
        target, mask = self._get_target_and_mask(bands)
        
        if allowed_materials is None:
            allowed_materials = list(range(self.n_materials))
            
        n_allowed = len(allowed_materials)
        B = n_restarts * n_allowed
        target = target.expand(B, -1)
        mask = mask.expand(B, -1)
        
        # Profile roughly in nm (e.g. 0 to 500)
        profile = torch.rand(B, self.nx, device=self.device) * 500.0
        h = torch.rand(B, 1, device=self.device) * (self.h_max - self.h_min) + self.h_min
        inc_ang = torch.rand(B, 1, device=self.device) * (self.inc_max - self.inc_min) + self.inc_min
        
        profile = nn.Parameter(profile)
        h = nn.Parameter(h)
        inc_ang = nn.Parameter(inc_ang)
        
        mat = torch.tensor(allowed_materials, device=self.device).repeat_interleave(n_restarts)
        
        # Note: lr is much higher because profile values are around ~0-1000 nm, while geo angles are ~0-2pi
        optimizer = optim.Adam([profile, h, inc_ang], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        history = torch.zeros((steps, n_allowed), device=self.device)
        pbar = tqdm(range(steps), desc="Optimizing Profile", disable=not show_progress)
        for step in pbar:
            optimizer.zero_grad()
            pred = self.forward_model(profile=profile, h=h, inc_ang=inc_ang, material_id=mat)
                
            abs_vals = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
            loss = -abs_vals
            loss_sum = loss.mean()
            loss_sum.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                # Enforce DC term = sum(A_n) constraint to match torcwa's geometry parameterization
                P_fft = torch.fft.rfft(profile, dim=1)
                amps = 2.0 * P_fft.abs() / self.nx
                n_harmonics = self.nx // 2
                A_n = amps[:, 1:n_harmonics+1]
                target_mean = A_n.sum(dim=1, keepdim=True)
                
                current_mean = profile.mean(dim=1, keepdim=True)
                profile.add_(target_mean - current_mean)
                
                profile.clamp_(min=0.0) # Physical constraint
                h.clamp_(self.h_min, self.h_max)
                inc_ang.clamp_(self.inc_min, self.inc_max)
                
                avg_abs = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
                abs_reshaped = avg_abs.view(n_allowed, n_restarts)
                max_vals = abs_reshaped.max(dim=1).values
                history[step] = max_vals
                from Utils.models import MATERIAL_LIBRARY
                postfix_dict = {
                    list(MATERIAL_LIBRARY.keys())[mat_idx]: f"{max_vals[i].item():.4f}" 
                    for i, mat_idx in enumerate(allowed_materials)
                }
                pbar.set_postfix(postfix_dict)
                
        with torch.no_grad():
            pred = self.forward_model(profile=profile, h=h, inc_ang=inc_ang, material_id=mat)
            final_abs = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
            
        final_abs_reshaped = final_abs.view(n_allowed, n_restarts)
        top_vals, top_indices_local = final_abs_reshaped.topk(top_k, dim=1, largest=True)
        
        top_results = []
        for i in range(n_allowed):
            mat_idx = allowed_materials[i]
            for k in range(top_k):
                local_idx = top_indices_local[i, k].item()
                global_idx = i * n_restarts + local_idx
                top_results.append({
                    "material_idx": mat_idx,
                    "profile": profile[global_idx].detach().cpu(),
                    "h": h[global_idx].detach().cpu(),
                    "inc_ang": inc_ang[global_idx].detach().cpu(),
                    "curve": pred[global_idx].detach().float().cpu(),
                    "loss": final_abs[global_idx].item()
                })
                
        best_global_idx = final_abs.argmax().item()
        
        return {
            "best_loss": final_abs[best_global_idx].item(),
            "best_profile": profile[best_global_idx].detach().cpu(),
            "best_h": h[best_global_idx].detach().cpu(),
            "best_inc_ang": inc_ang[best_global_idx].detach().cpu(),
            "best_material": mat[best_global_idx].item(),
            "best_curve": pred[best_global_idx].detach().float().cpu(),
            "target": target[0].float().cpu(),
            "mask": mask[0].float().cpu(),
            "mode": "profile",
            "history": history.cpu(),
            "top_results": top_results,
            "allowed_materials": allowed_materials
        }

    def optimize_de(self, bands: List[Tuple[float, float]], pop_size: int = 10000, generations: int = 300, 
                    F: float = 0.8, CR: float = 0.9, allowed_materials: list[int] = None, top_k: int = 2, show_progress: bool = True) -> dict:
        self.forward_model.eval()
        target, mask = self._get_target_and_mask(bands)
        
        if allowed_materials is None:
            allowed_materials = list(range(self.n_materials))
            
        n_allowed = len(allowed_materials)
        B = pop_size * n_allowed
        target = target.expand(B, -1)
        mask = mask.expand(B, -1)
        
        pop = torch.rand(B, len(self.geo_min), device=self.device) * (self.geo_max - self.geo_min) + self.geo_min
        mat = torch.tensor(allowed_materials, device=self.device).repeat_interleave(pop_size)
        
        with torch.no_grad():
            pred = self.forward_model(geometry=pop, material_id=mat)
            fitness = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
            
            range_geo = self.geo_max - self.geo_min
            active_mask = (range_geo > 1e-5).float()
            
            p_norm = (pop - self.geo_min) / (range_geo + 1e-9)
            boundary_penalty = 0.05 * torch.sum(active_mask * torch.pow((p_norm - 0.5) * 2.0, 10), dim=-1)
            fitness = fitness - boundary_penalty
            
        history = torch.zeros((generations, n_allowed), device=self.device)
        
        pbar = tqdm(range(generations), desc="Optimizing DE", disable=not show_progress, leave=False)
        for gen in pbar:
            # Learning Rate Schedule: High F for exploration, drop by half for the last 33% of generations to polish local minima
            current_F = F if gen < (generations * 2 / 3) else F / 2.0
            
            pop_reshaped = pop.view(n_allowed, pop_size, -1)
            
            r1 = torch.randint(0, pop_size, (n_allowed, pop_size), device=self.device)
            r2 = torch.randint(0, pop_size, (n_allowed, pop_size), device=self.device)
            r3 = torch.randint(0, pop_size, (n_allowed, pop_size), device=self.device)
            
            a = pop_reshaped.gather(1, r1.unsqueeze(-1).expand_as(pop_reshaped))
            b = pop_reshaped.gather(1, r2.unsqueeze(-1).expand_as(pop_reshaped))
            c = pop_reshaped.gather(1, r3.unsqueeze(-1).expand_as(pop_reshaped))
            
            mutant = a + current_F * (b - c)
            mutant = torch.clamp(mutant, self.geo_min, self.geo_max)
            
            # Explicitly force constants
            if getattr(self, "h_val", None) is not None:
                mutant[..., -2] = self.h_val
            if getattr(self, "inc_val", None) is not None:
                mutant[..., -1] = self.inc_val
            
            cross_mask = torch.rand_like(mutant) < CR
            force_mut_idx = torch.randint(0, mutant.shape[-1], (n_allowed, pop_size, 1), device=self.device)
            cross_mask.scatter_(-1, force_mut_idx, True)
            
            trial = torch.where(cross_mask, mutant, pop_reshaped).view(B, -1)
            
            with torch.no_grad():
                trial_pred = self.forward_model(geometry=trial, material_id=mat)
                trial_fitness = (trial_pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
                
                trial_p_norm = (trial - self.geo_min) / (range_geo + 1e-9)
                trial_penalty = 0.05 * torch.sum(active_mask * torch.pow((trial_p_norm - 0.5) * 2.0, 10), dim=-1)
                trial_fitness = trial_fitness - trial_penalty
                
                improve = trial_fitness > fitness
                pop[improve] = trial[improve]
                fitness[improve] = trial_fitness[improve]
                pred[improve] = trial_pred[improve]
                
                fitness_reshaped = fitness.view(n_allowed, pop_size)
                max_indices = fitness_reshaped.argmax(dim=1)
                
                # We need pure absorptance from the current population
                # Note: fitness has the penalty permanently subtracted. We recalculate pure abs
                # Wait, fitness is persistently penalized, so we must calculate pure abs directly from pred
                pure_abs = (pred.float() * mask).sum(dim=-1) / mask.sum(dim=-1)
                pure_reshaped = pure_abs.view(n_allowed, pop_size)
                
                best_pure_abs = pure_reshaped[torch.arange(n_allowed), max_indices]
                history[gen] = best_pure_abs
                
                from Utils.models import MATERIAL_LIBRARY
                postfix_dict = {
                    list(MATERIAL_LIBRARY.keys())[mat_idx]: f"{best_pure_abs[i].item():.4f}" 
                    for i, mat_idx in enumerate(allowed_materials)
                }
                pbar.set_postfix(postfix_dict)
                
        fitness_reshaped = fitness.view(n_allowed, pop_size)
        top_vals, top_indices_local = fitness_reshaped.topk(top_k, dim=1, largest=True)
        
        top_results = []
        for i in range(n_allowed):
            mat_idx = allowed_materials[i]
            for k in range(top_k):
                local_idx = top_indices_local[i, k].item()
                global_idx = i * pop_size + local_idx
                top_results.append({
                    "material_idx": mat_idx,
                    "geometry": pop[global_idx].detach().cpu(),
                    "curve": pred[global_idx].detach().float().cpu(),
                    "loss": fitness[global_idx].item()
                })
                
        best_global_idx = fitness.argmax().item()
        
        return {
            "best_loss": fitness[best_global_idx].item(),
            "best_geometry": pop[best_global_idx].detach().cpu(),
            "best_material": mat[best_global_idx].item(),
            "best_curve": pred[best_global_idx].detach().float().cpu(),
            "target": target[0].float().cpu(),
            "mask": mask[0].float().cpu(),
            "mode": "de",
            "history": history.cpu(),
            "top_results": top_results,
            "allowed_materials": allowed_materials
        }

def recover_geometry_from_profile(profile_1d: torch.Tensor, h: torch.Tensor, inc_ang: torch.Tensor, nx: int = 128, n_harmonics: Optional[int] = None) -> torch.Tensor:
    """
    Given a spatial profile, recover harmonic geometry parameters via FFT.
    If n_harmonics is None, recovers all spatial resolution harmonics (nx // 2).
    Returns: geometry tensor of shape [n_harmonics * 2 + 2]
    """
    P_fft = torch.fft.rfft(profile_1d)
    
    amps = 2.0 * P_fft.abs() / nx
    
    if n_harmonics is None:
        n_harmonics = nx // 2
    else:
        n_harmonics = min(n_harmonics, nx // 2)
        
    A_n = amps[1:n_harmonics+1]
    phi_n = (-P_fft[1:n_harmonics+1].angle()) % (2 * torch.pi)
    
    # Sanity check: Ensure DC term matches the sum of AC amplitudes
    dc_component = profile_1d.mean(dim=-1) if profile_1d.dim() > 1 else profile_1d.mean()
    sum_A_n = A_n.sum(dim=-1) if A_n.dim() > 1 else A_n.sum()
    
    if not torch.allclose(dc_component, sum_A_n, atol=1e-3):
        print(f"Warning: DC sanity check failed! profile.mean = {dc_component.mean().item():.5f}, sum(A_n) = {sum_A_n.mean().item():.5f}")
        
    geo = torch.zeros(n_harmonics * 2 + 2, device=profile_1d.device)
    geo[0:n_harmonics*2:2] = A_n
    geo[1:n_harmonics*2:2] = phi_n
    geo[-2] = h.squeeze()
    geo[-1] = inc_ang.squeeze()
    
    return geo
