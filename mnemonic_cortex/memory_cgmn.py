import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import fast_pairwise_l2

class EnhancedCGMNMemory(nn.Module):
    """Curved Geometric Memory Network (CGMN) with safe writes and curvature-aware attention.
    """
    def __init__(self, input_dim: int, manifold_dim: int = 16, mem_slots: int = 512,
                 slot_dim: int = 256, topk: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.D, self.M, self.H = manifold_dim, mem_slots, slot_dim
        self.K = min(topk, mem_slots)

        self.manifold_projection = nn.Sequential(
            nn.Linear(input_dim, self.D * 3),
            nn.LayerNorm(self.D * 3),
            nn.GELU()
        )
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.positional_encoding = nn.Parameter(torch.randn(self.M, self.D, 3))
        self.curvature = nn.Parameter(torch.randn(self.M, self.D))
        self.curv_alpha = nn.Parameter(torch.tensor(0.1))

        # Simple ODE dynamics in manifold space (Euler steps, keeps deps minimal)
        self.ode_dynamics = nn.Sequential(nn.Linear(self.D * 3, 128), nn.Tanh(), nn.Linear(128, self.D * 3))
        self.ode_steps = 2
        self.ode_dt = 0.5

        self.output_projection = nn.Sequential(nn.Linear(self.H, input_dim), nn.LayerNorm(input_dim), nn.GELU())

        # Temperature (for attention distance scaling)
        self.register_buffer("temperature", torch.tensor(1.0))

    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
            self.temperature = t.mean().detach()

    def manifold_ode_step(self, x):
        # x: (B,S,D,3)
        B,S,D,_ = x.shape
        x_flat = x.view(B,S,-1)
        dx = self.ode_dynamics(x_flat).view_as(x)
        return x + self.ode_dt * dx

    def _evolve(self, man):
        x = man
        for _ in range(self.ode_steps):
            x = self.manifold_ode_step(x)
        return x

    def _attend(self, query, positions):
        # query: (B,S,D) ; positions: (B,S,D,3)
        mem_pos = self.positional_encoding.view(self.M, -1)     # (M,3D)
        q = positions.flatten(2).detach()                       # (B,S,3D)
        dist = fast_pairwise_l2(q, mem_pos)                     # (B,S,M)

        # Temperature scaling
        dist = dist / self.temperature.clamp_min(1e-6)

        curv_w = torch.exp(-self.curv_alpha * self.curvature.norm(dim=-1))  # (M,)
        dist = dist * curv_w.view(1,1,self.M)
        dtop, itop = torch.topk(dist, self.K, dim=-1, largest=False)        # (B,S,K)
        w = F.softmax(-dtop, dim=-1)                                        # (B,S,K)
        mem = self.memory_slots[itop]                                       # (B,S,K,H)
        attended = torch.sum(w.unsqueeze(-1) * mem, dim=2)                  # (B,S,H)
        return attended, (w, itop)

    @torch.no_grad()
    def _write(self, encoded, weights_indices, ema=0.9):
        # encoded: (B,S,H); weights_indices=(w:(B,S,K), idx:(B,S,K))
        w, idx = weights_indices
        B,S,K = idx.shape
        updates = torch.sum(w.unsqueeze(-1) * encoded.unsqueeze(2), dim=1)  # (B,K,H)

        flat_idx = idx.reshape(-1)                                          # (B*K,)
        flat_upd = updates.reshape(-1, self.H)                              
        accum = torch.zeros_like(self.memory_slots)
        accum.index_add_(0, flat_idx, flat_upd)                             # sum updates per slot
        counts = torch.zeros(self.M, device=accum.device).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=accum.dtype))
        counts = counts.clamp_min_(1.0).unsqueeze(-1)
        avg_upd = accum / counts
        self.memory_slots.data.mul_(ema).add_((1-ema)*avg_upd)

    def forward(self, x, operation='read'):
        B,S,_ = x.shape
        man = self.manifold_projection(x).view(B,S,self.D,3)
        evolved = self._evolve(man)
        query = evolved.mean(dim=3)                                  # (B,S,D)
        attended, wi = self._attend(query, evolved)
        if operation == 'write':
            enc = attended                                           # (B,S,H) as a simple writer proxy
            self._write(enc, wi)
            return x
        return self.output_projection(attended)                      # (B,S,input_dim)
