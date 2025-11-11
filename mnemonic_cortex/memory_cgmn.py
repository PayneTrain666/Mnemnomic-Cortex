import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .utils import fast_pairwise_l2

class EnhancedCGMNMemory(nn.Module):
    """Curved Geometric Memory Network (CGMN) with lightbulb-aware temperature & plasticity."""
    def __init__(self, input_dim: int, manifold_dim: int = 16, mem_slots: int = 512,
                 slot_dim: int = 256, topk: int = 32, use_per_sample_temp: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.D, self.M, self.H = manifold_dim, mem_slots, slot_dim
        self.K_base = min(topk, mem_slots)
        self.use_per_sample_temp = use_per_sample_temp
        self.temp_variance_threshold = 0.3  # fallback to scalar if variance > this

        self.manifold_projection = nn.Sequential(
            nn.Linear(input_dim, self.D * 3),
            nn.LayerNorm(self.D * 3),
            nn.GELU()
        )
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.positional_encoding = nn.Parameter(torch.randn(self.M, self.D, 3))
        self.curvature = nn.Parameter(torch.randn(self.M, self.D))
        self.curv_alpha = nn.Parameter(torch.tensor(0.1))

        # Simple ODE dynamics in manifold space
        self.ode_dynamics = nn.Sequential(nn.Linear(self.D * 3, 128), nn.Tanh(), nn.Linear(128, self.D * 3))
        self.ode_steps = 2
        self.ode_dt = 0.5

        self.output_projection = nn.Sequential(nn.Linear(self.H, input_dim), nn.LayerNorm(input_dim), nn.GELU())

        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M

        # internal lightbulb tracker
        self.register_buffer('lb_top1_avg', torch.tensor(1.0))
        self.lb_momentum = 0.99
        self.lb_drop_ratio = 0.7

    # ---------------- Controls ----------------
    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        elif self.use_per_sample_temp:
            # Per-sample temperature with variance check
            if t.numel() > 1:
                variance = t.var()
                if variance > self.temp_variance_threshold:
                    # Too unstable, fall back to mean
                    self.temperature = t.mean().detach()
                else:
                    # Keep per-sample temps
                    self.temperature = t.detach()
            else:
                self.temperature = t.detach()
        else:
            # Default: collapse to scalar
            self.temperature = t.mean().detach()

    def set_active_fraction(self, frac: float):
        frac = float(max(0.1, min(1.0, frac)))
        self.active_slots = max(8, int(self.M * frac))

    def enable_energy_efficient_mode(self, enable: bool = True):
        self.set_active_fraction(0.5 if enable else 1.0)

    # ---------- DDP sync -------------
    @torch.no_grad()
    def _sync_buffers_ddp(self):
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        for t in [self.memory_slots.data, self.usage_counts]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1, ema: float = 0.9):
        if self.usage_counts.max() <= 0:
            return
        usage_ratio = self.usage_counts / (self.usage_counts.max() + 1e-6)
        mask = usage_ratio < threshold
        if mask.any():
            mean_slot = self.memory_slots[~mask].mean(dim=0, keepdim=True) if (~mask).any() else self.memory_slots.mean(dim=0, keepdim=True)
            self.memory_slots.data[mask] = ema * self.memory_slots.data[mask] + (1-ema) * mean_slot
            self.usage_counts[mask] = 0.0

    def get_metrics(self):
        """Return dict of diagnostic metrics."""
        M = self.active_slots
        return {
            'cgmn_temp': self.temperature.mean().item() if self.temperature.numel() > 1 else self.temperature.item(),
            'cgmn_topk_base': self.K_base,
            'cgmn_active_slots': M,
            'cgmn_usage_mean': self.usage_counts[:M].mean().item(),
            'cgmn_usage_max': self.usage_counts[:M].max().item(),
            'cgmn_lb_top1_avg': self.lb_top1_avg.item(),
        }

    # ---------------- Core ----------------
    def manifold_ode_step(self, x):
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
        M = self.active_slots
        mem_pos = self.positional_encoding[:M].view(M, -1)     # (M,3D)
        q = positions.flatten(2).detach()                      # (B,S,3D)
        dist = fast_pairwise_l2(q, mem_pos)                    # (B,S,M)

        # Temperature scaling + curvature weight
        # Handle both scalar and per-sample temp
        if self.temperature.numel() == 1:
            temp_scale = self.temperature.clamp_min(1e-6)
        else:
            # Per-sample: (B,) -> (B,1,1) for broadcasting
            temp_scale = self.temperature.view(-1, 1, 1).clamp_min(1e-6)
        dist = dist / temp_scale
        curv_w = torch.exp(-self.curv_alpha * self.curvature[:M].norm(dim=-1))  # (M,)
        dist = dist * curv_w.view(1,1,M)

        # Top-k with internal lightbulb boost
        K = min(self.K_base, M)
        dtmp, _ = torch.topk(dist, max(1,K), dim=-1, largest=False)
        top1 = dtmp[...,0].mean(dim=1).mean(dim=0) if dtmp.dim()==3 else dtmp.mean()
        self.lb_top1_avg = self.lb_momentum * self.lb_top1_avg + (1-self.lb_momentum) * top1.detach()
        internal_fire = bool(top1 < self.lb_drop_ratio * self.lb_top1_avg)
        if internal_fire:
            K = min(M, max(K, int(self.K_base * 1.5)))

        dtop, itop = torch.topk(dist, K, dim=-1, largest=False)        # (B,S,K)
        w = torch.softmax(-dtop, dim=-1)                               # (B,S,K)
        mem = self.memory_slots[:M][itop]                              # (B,S,K,H)
        attended = torch.sum(w.unsqueeze(-1) * mem, dim=2)             # (B,S,H)
        return attended, (w, itop), internal_fire

    @torch.no_grad()
    def _record_usage(self, indices):
        flat = indices.reshape(-1)
        self.usage_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage_counts.dtype))

    @torch.no_grad()
    def _write(self, encoded, weights_indices, ema=0.9):
        w, idx = weights_indices
        B,S,K = idx.shape
        updates = torch.sum(w.unsqueeze(-1) * encoded.unsqueeze(2), dim=1)  # (B,K,H)
        flat_idx = idx.reshape(-1)
        flat_upd = updates.reshape(-1, self.H)
        accum = torch.zeros_like(self.memory_slots)
        accum.index_add_(0, flat_idx, flat_upd)
        counts = torch.zeros(self.M, device=accum.device).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=accum.dtype))
        counts = counts.clamp_min_(1.0).unsqueeze(-1)
        avg_upd = accum / counts
        self.memory_slots.data.mul_(ema).add_((1-ema)*avg_upd)
        # DDP sync
        self._sync_buffers_ddp()

    def forward(self, x, operation='read', fire_mask=None, recall_boost: float = 0.3):
        """
        fire_mask: optional (B,) bool Tensor. If any True, we sharpen attention by lowering temperature
                   and we also write with a lower EMA (more plastic).
        """
        B,S,_ = x.shape

        # --- Lower temperature if any batch fires ---
        saved_temp = self.temperature.clone()
        any_ext_fire = (
            isinstance(fire_mask, torch.Tensor) and fire_mask.any()
        ) or (
            isinstance(fire_mask, bool) and fire_mask
        )
        if any_ext_fire:
            self.temperature = (self.temperature / (1.0 + recall_boost)).detach()

        man = self.manifold_projection(x).view(B,S,self.D,3)
        evolved = self._evolve(man)
        query = evolved.mean(dim=3)                                  # (B,S,D)
        attended, wi, internal_fire = self._attend(query, evolved)
        self._record_usage(wi[1])

        if operation == 'write':
            # More plastic on fire
            ema = 0.85 if (internal_fire or any_ext_fire) else 0.90
            enc = attended                                           # (B,S,H)
            self._write(enc, wi, ema=ema)
            self.temperature = saved_temp
            return x

        out = self.output_projection(attended)                      # (B,S,input_dim)
        self.temperature = saved_temp
        return out
