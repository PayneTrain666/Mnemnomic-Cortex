import torch
import torch.nn as nn

class EnhancedSensoryBuffer(nn.Module):
    """Simple sensory buffer with self-attention + GRU and salience blending.
    Maintains a small cache of pooled summaries.
    """
    def __init__(self, buffer_size=5, input_dim=512):
        super().__init__()
        self.buffer_size = buffer_size
        self.input_dim = input_dim
        self.num_heads = self._pick_num_heads(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, self.num_heads, batch_first=True)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.salience = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
        self.norm = nn.LayerNorm(input_dim)
        from collections import deque
        self._cache = deque(maxlen=buffer_size)  # store recent salience scores

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if int(dim) % h == 0:
                return h
        return 1

    def _salience_score(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # (B,d)
        return self.salience(pooled).mean().clamp(0.1, 1.0)

    def update(self, x):  # x: (B,S,d)
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x [B,S,{self.input_dim}], got {tuple(x.shape)}")
        s = float(self._salience_score(x).detach().item())
        self._cache.append(s)

    def attention_filter(self, x):  # x: (B,S,d) -> (B,S,d)
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x [B,S,{self.input_dim}], got {tuple(x.shape)}")
        h, _ = self.gru(x)                                # (B,S,d)
        y, _ = self.attn(h, h, h)
        current = self._salience_score(h)
        if len(self._cache) > 0:
            hist = torch.tensor(sum(self._cache) / len(self._cache), device=x.device, dtype=x.dtype)
            sal = (0.7 * current) + (0.3 * hist)
        else:
            sal = current
        out = sal * y + (1.0 - sal) * x
        return self.norm(out)
