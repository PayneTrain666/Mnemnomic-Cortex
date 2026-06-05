import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.08
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


def build_optimizer(params: Iterable, cfg: OptimizerConfig) -> Optimizer:
    trainable = [p for p in params if getattr(p, "requires_grad", False)]
    name = str(cfg.name).strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            trainable,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            betas=tuple(cfg.betas),
            eps=float(cfg.eps),
        )
    if name == "adam":
        return torch.optim.Adam(
            trainable,
            lr=float(cfg.lr),
            betas=tuple(cfg.betas),
            eps=float(cfg.eps),
        )
    raise ValueError(f"unknown optimizer name={cfg.name}")


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.08,
) -> LambdaLR:
    total = max(1, int(total_steps))
    warmup = max(0, int(total * float(warmup_ratio)))
    floor = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        s = int(step) + 1
        if warmup > 0 and s <= warmup:
            return max(floor, s / float(warmup))
        if total <= warmup:
            return floor
        progress = min(1.0, max(0.0, (s - warmup) / float(max(1, total - warmup))))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda)


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
