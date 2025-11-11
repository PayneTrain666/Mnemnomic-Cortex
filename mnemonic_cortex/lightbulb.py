import torch
import torch.nn as nn

class LightbulbDetector(nn.Module):
    """Detects 'lightbulb' moments: spikes in an internal salience score.
    Returns a boolean mask per batch indicating a trigger.
    Includes adaptive threshold calibration to maintain target trigger rate.
    """
    def __init__(self, dim: int, thresh: float = 2.0, target_rate: float = 0.10, adapt_lr: float = 0.01):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.thresh = thresh
        self.target_rate = target_rate  # target trigger rate (e.g., 10%)
        self.adapt_lr = adapt_lr        # learning rate for threshold adaptation
        
        # Tracking buffers
        self.register_buffer('trigger_rate_ema', torch.tensor(0.0))
        self.register_buffer('total_samples', torch.tensor(0))
        self.register_buffer('total_fires', torch.tensor(0))
        self.momentum = 0.99

    def forward(self, x: torch.Tensor):
        # x: (B,S,d)
        B = x.size(0)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            s = self.proj(x).squeeze(-1)  # (B,S)
            mean = s.mean(dim=1, keepdim=True)
            std  = s.std(dim=1, keepdim=True).clamp_min(1e-6)
            z = (s - mean) / std
            fire = z.max(dim=1).values > self.thresh  # (B,)
        
        # Update tracking
        if self.training:
            with torch.no_grad():
                batch_rate = fire.float().mean()
                self.trigger_rate_ema = self.momentum * self.trigger_rate_ema + (1 - self.momentum) * batch_rate
                self.total_samples += B
                self.total_fires += fire.sum()
                
                # Adaptive threshold adjustment
                if self.trigger_rate_ema > self.target_rate + 0.02:
                    # Too many triggers, increase threshold
                    self.thresh += self.adapt_lr
                elif self.trigger_rate_ema < self.target_rate - 0.02:
                    # Too few triggers, decrease threshold
                    self.thresh = max(0.5, self.thresh - self.adapt_lr)
        
        return fire
    
    def get_metrics(self):
        """Return dict of current lightbulb metrics."""
        return {
            'trigger_rate_ema': self.trigger_rate_ema.item(),
            'threshold': self.thresh,
            'total_fires': self.total_fires.item(),
            'total_samples': self.total_samples.item(),
        }

class ExplosiveRecallScaler(nn.Module):
    """If a lightbulb fires, reduce distance temperature (i.e., sharpen attention).
    """
    def __init__(self, base_temp: float = 1.0, min_temp: float = 0.5, boost: float = 0.2):
        super().__init__()
        self.base_temp = base_temp
        self.min_temp = min_temp
        self.boost = boost

    def forward(self, fire_mask: torch.Tensor):
        temp = torch.full_like(fire_mask, self.base_temp, dtype=torch.float32)
        temp = temp - fire_mask.to(temp.dtype) * self.boost
        return temp.clamp(self.min_temp, self.base_temp + 1.0)
