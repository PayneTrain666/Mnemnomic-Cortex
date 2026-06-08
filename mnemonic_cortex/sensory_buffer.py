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
        self.num_heads = self._pick_num_heads(int(input_dim))
        self.attn = nn.MultiheadAttention(input_dim, self.num_heads, batch_first=True)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.salience = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
        from collections import deque
        self._cache = deque(maxlen=buffer_size)  # store pooled summaries only

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    def update(self, x):  # x: (B,S,d)
        with torch.no_grad():
            pooled = x.mean(dim=1)                        # (B,d)
            s = self.salience(pooled).mean()             # scalar salience
        self._cache.append((pooled, s))

    def attention_filter(self, x):  # x: (B,S,d) -> (B,S,d)
        h, _ = self.gru(x)                                # (B,S,d)
        y, _ = self.attn(h, h, h)
        if len(self._cache) == 0:
            return y
        _, s = zip(*self._cache)
        sal = torch.stack(s).mean().clamp(0.1, 1.0).item()
        return sal * y + (1 - sal) * x
