import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L] # causal padding
        x = rearrange(x, 'b d l -> b l d')

        x = F.silu(x)

        x_dbl = self.x_proj(x) # (B, L, d_state * 2 + 1)
        delta, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta)) # (B, L, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Discretize
        # dA = exp(delta * A) -> (B, L, d_inner, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        # dB = delta * B -> (B, L, d_inner, d_state)
        dB = torch.einsum('bld,bln->bldn', delta, B_mat)

        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C_mat[:, i])
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (B, L, d_inner)
        y = y + x * self.D
        
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

class MambaNet(nn.Module):
    def __init__(self, d_input, d_model=64, n_layers=4):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, d_input)
        x = self.embedding(x)
        for layer in self.layers:
            # Add residual connection
            x = x + layer(x)
        return self.norm_f(x)
