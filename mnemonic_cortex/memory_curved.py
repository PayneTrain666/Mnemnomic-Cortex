import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class EnhancedCurvedMemory(nn.Module):
    """A simpler curved memory with scalar curvature gating and associative weights.
    Added usage tracking, active slots, consolidation, energy mode.
    Functions as working memory as well.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, curvature_dim: int = 8, mem_slots: int = 128, topk: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.H, self.M = hidden_dim, mem_slots
        self.K_base = min(topk, mem_slots)
        self.encoder = nn.Sequential(nn.Linear(input_dim, self.H), nn.Tanh())
        self.curvature = nn.Parameter(torch.randn(curvature_dim))
        self.curv_proj = nn.Linear(self.H, 1)  # scalar gate
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.memory_importance = nn.Parameter(torch.ones(self.M))
        self.associative_weights = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        self.decoder = nn.Sequential(nn.Linear(self.H, input_dim), nn.Tanh())

        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M

    # Controls
    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
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
        for t in [self.memory_slots.data, self.usage_counts, self.associative_weights.data]:
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
            'curved_temp': self.temperature.mean().item() if self.temperature.numel() > 1 else self.temperature.item(),
            'curved_topk_base': self.K_base,
            'curved_active_slots': M,
            'curved_usage_mean': self.usage_counts[:M].mean().item(),
            'curved_usage_max': self.usage_counts[:M].max().item(),
            'curved_importance_mean': self.memory_importance[:M].mean().item(),
        }

    # Core ops
    def content_based_addressing(self, query):  # query: (B,H)
        M = self.active_slots
        sim = torch.einsum('bd,md->bm', query, self.memory_slots[:M])      # (B,M)
        gate = torch.sigmoid(self.curv_proj(query))                        # (B,1)
        sim = sim * (0.5 + gate)                                           # scalar gating
        sim = sim / self.temperature.clamp_min(1e-6)
        K = min(self.K_base, M)
        vals, idx = torch.topk(sim, K, dim=-1)
        return vals, idx

    @torch.no_grad()
    def update_memory(self, x, importance, indices):
        enc = self.encoder(x).mean(dim=1)                 # (B,H)
        B,K = indices.shape
        mem = self.memory_slots[indices]                  # (B,K,H)
        gate = torch.sigmoid(self.memory_importance[indices].unsqueeze(-1))  # (B,K,1)
        cand = enc.unsqueeze(1)                           # (B,1,H)
        upd = gate * mem + (1-gate) * cand                # (B,K,H)
        flat_idx = indices.reshape(-1)
        flat_upd = upd.reshape(-1, self.H)
        accum = torch.zeros_like(self.memory_slots)
        accum.index_add_(0, flat_idx, flat_upd)
        counts = torch.zeros(self.M, device=accum.device).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=accum.dtype))
        counts = counts.clamp_min_(1.0).unsqueeze(-1)
        avg_upd = accum / counts
        self.memory_slots.data.mul_(0.95).add_(0.05 * avg_upd)
        self._sync_buffers_ddp()
        # importance EMA (crude)
        self.memory_importance.data.index_add_(0, flat_idx, 0.05 * torch.ones_like(flat_idx, dtype=self.memory_importance.dtype).to(self.memory_importance.device))

    @torch.no_grad()
    def _record_usage(self, indices):
        flat = indices.reshape(-1)
        self.usage_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage_counts.dtype))

    def associative_activation(self, query):  # (B,H) -> (B,M)
        M = self.active_slots
        act = torch.einsum('bd,md->bm', query, self.memory_slots[:M])
        for _ in range(2):
            act = torch.softmax(act, dim=-1)
            act = torch.einsum('bm,mn->bn', act, self.associative_weights[:M,:M])
        return act

    def forward(self, x, operation='read', importance=None):
        enc = self.encoder(x)                             # (B,S,H)
        query = enc.mean(dim=1)                           # (B,H)
        if operation == 'write':
            _, idx = self.content_based_addressing(query)
            self._record_usage(idx)
            self.update_memory(x, importance, idx)
            return x
        vals, idx = self.content_based_addressing(query)  # (B,K)
        self._record_usage(idx)
        act = self.associative_activation(query)          # (B,M_active)
        comb = torch.softmax(vals + act.gather(1, idx), dim=-1)  # (B,K)
        mem = self.memory_slots[idx]                      # (B,K,H)
        read = torch.sum(comb.unsqueeze(-1) * mem, dim=1) # (B,H)
        return self.decoder(read).unsqueeze(1).expand(-1, x.size(1), -1)  # (B,S,input_dim)
