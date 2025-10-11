import torch
import torch.nn as nn

class EnhancedSensoryBuffer(nn.Module):
    """Simple 3-D sensory buffer with self-attention + GRU and salience blending.
    Maintains a small cache of pooled summaries.
    """
    def __init__(self, buffer_size=5, input_dim=512):
        super().__init__()
        self.buffer_size = buffer_size
        self.input_dim = input_dim
        self.attn = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.salience = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
        from collections import deque
        self._cache = deque(maxlen=buffer_size)  # store pooled summaries only

    def update(self, x):  # x: (B,S,d)
        with torch.no_grad():
            pooled = x.mean(dim=1)                        # (B,d)
            s = self.salience(pooled)                     # (B,1)
        self._cache.append((pooled, s))

    def attention_filter(self, x):  # x: (B,S,d) -> (B,S,d)
        h, _ = self.gru(x)                                # (B,S,d)
        y, _ = self.attn(h, h, h)
        if len(self._cache) == 0:
            return y
        _, s = zip(*self._cache)
        sal = torch.stack(s).mean().clamp(0.1, 1.0).item()
        return sal * y + (1 - sal) * x
