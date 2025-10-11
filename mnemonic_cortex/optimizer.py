import time
import torch
import torch.nn.functional as F

class MemoryOptimizer:
    """Profiles memory stack performance (latency/energy-ish/accuracy proxy)."""
    def __init__(self, cortex):
        self.cortex = cortex
        self.metrics = { 'access_time': [], 'energy_usage': [], 'mse_proxy': [] }

    @torch.no_grad()
    def profile(self, test_batches):
        self.cortex.eval()
        for x, ctx in test_batches:
            start = time.time()
            out = self.cortex(x, ctx, operation='retrieve')
            dt = time.time() - start
            energy = sum(p.numel() for p in self.cortex.parameters()) / 1e6
            target = ctx  # pretend target
            mse = F.mse_loss(out, target[:out.size(0)], reduction='mean').item()
            self.metrics['access_time'].append(dt)
            self.metrics['energy_usage'].append(energy)
            self.metrics['mse_proxy'].append(mse)
        return {k: float(sum(v)/max(1,len(v))) for k,v in self.metrics.items()}

    def enable_energy_mode_if_slow(self, max_time: float = 0.1):
        avg = float(sum(self.metrics['access_time'])/max(1,len(self.metrics['access_time'])))
        if avg > max_time:
            self.cortex.enable_energy_mode(True)
            return True
        return False
