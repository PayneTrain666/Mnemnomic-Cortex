import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCurvedMemory(nn.Module):
    """A simpler curved memory with scalar curvature gating and associative weights.
    Functions as working memory as well.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, curvature_dim: int = 8, mem_slots: int = 128, topk: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.H, self.M = hidden_dim, mem_slots
        self.K = min(topk, mem_slots)
        self.encoder = nn.Sequential(nn.Linear(input_dim, self.H), nn.Tanh())
        self.curvature = nn.Parameter(torch.randn(curvature_dim))
        self.curv_proj = nn.Linear(self.H, 1)  # scalar gate
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.memory_importance = nn.Parameter(torch.ones(self.M))
        self.associative_weights = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        self.decoder = nn.Sequential(nn.Linear(self.H, input_dim), nn.Tanh())

        self.register_buffer("temperature", torch.tensor(1.0))

    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
            self.temperature = t.mean().detach()

    def content_based_addressing(self, query):  # query: (B,H)
        sim = torch.einsum('bd,md->bm', query, self.memory_slots)      # (B,M)
        gate = torch.sigmoid(self.curv_proj(query))                    # (B,1)
        sim = sim * (0.5 + gate)                                       # scalar gating
        sim = sim / self.temperature.clamp_min(1e-6)
        vals, idx = torch.topk(sim, self.K, dim=-1)
        return vals, idx

    @torch.no_grad()
    def update_memory(self, x, importance, indices):
        # x: (B,S,input_dim), importance: scalar or (B,1)
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
        # importance EMA (crude)
        self.memory_importance.data.index_add_(0, flat_idx, 0.05 * torch.ones_like(flat_idx, dtype=self.memory_importance.dtype).to(self.memory_importance.device))

    def associative_activation(self, query):  # (B,H) -> (B,M)
        act = torch.einsum('bd,md->bm', query, self.memory_slots)
        for _ in range(2):
            act = torch.softmax(act, dim=-1)
            act = torch.einsum('bm,mn->bn', act, self.associative_weights)
        return act

    def forward(self, x, operation='read', importance=None):
        enc = self.encoder(x)                             # (B,S,H)
        query = enc.mean(dim=1)                           # (B,H)
        if operation == 'write':
            _, idx = self.content_based_addressing(query)
            self.update_memory(x, importance, idx)
            return x
        vals, idx = self.content_based_addressing(query)  # (B,K)
        act = self.associative_activation(query)          # (B,M)
        comb = torch.softmax(vals + act.gather(1, idx), dim=-1)  # (B,K)
        mem = self.memory_slots[idx]                      # (B,K,H)
        read = torch.sum(comb.unsqueeze(-1) * mem, dim=1) # (B,H)
        return self.decoder(read).unsqueeze(1).expand(-1, x.size(1), -1)  # (B,S,input_dim)
