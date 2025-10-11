import torch
import torch.nn as nn
import torch.nn.functional as F

class LightbulbDetector(nn.Module):
    """Detects 'lightbulb' moments: spikes in an internal salience score.
    Returns a boolean mask per batch indicating a trigger.
    """
    def __init__(self, dim: int, thresh: float = 2.0):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.thresh = thresh

    def forward(self, x: torch.Tensor):
        # x: (B,S,d)
        # Keep computation in FP32 regardless of outer autocast context
        with torch.cuda.amp.autocast(enabled=False):
            s = self.proj(x.float()).squeeze(-1)  # (B,S) in fp32
        mean = s.mean(dim=1, keepdim=True)
        std  = s.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = (s - mean) / std
        fire = z.max(dim=1).values > self.thresh  # (B,)
        return fire

class ExplosiveRecallScaler(nn.Module):
    """If a lightbulb fires, reduce distance temperature (i.e., sharpen attention).
    """
    def __init__(self, base_temp: float = 1.0, min_temp: float = 0.5, boost: float = 0.2):
        super().__init__()
        self.base_temp = base_temp
        self.min_temp = min_temp
        self.boost = boost

    def forward(self, fire_mask: torch.Tensor):
        # fire_mask: (B,) boolean
        # return per-batch temperature scalars: lower temp => sharper distributions
        temp = torch.full_like(fire_mask, self.base_temp, dtype=torch.float32)
        temp = temp - fire_mask.to(temp.dtype) * self.boost
        return temp.clamp(self.min_temp, self.base_temp + 1.0)
