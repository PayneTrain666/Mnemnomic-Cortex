from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    momentum: float = 0.9
    nesterov: bool = False
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1

    def validate(self) -> None:
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if float(self.grad_clip) <= 0.0:
            raise ValueError("grad_clip must be positive")
        if not (0.0 <= float(self.warmup_ratio) < 1.0):
            raise ValueError("warmup_ratio must be in [0,1)")
        if not (0.0 <= float(self.min_lr_ratio) <= 1.0):
            raise ValueError("min_lr_ratio must be in [0,1]")
        b1, b2 = self.betas
        if not (0.0 < float(b1) < 1.0 and 0.0 < float(b2) < 1.0):
            raise ValueError("betas must be in (0,1)")


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: Optional[OptimizerConfig] = None):
    cfg = cfg or OptimizerConfig()
    cfg.validate()
    name = str(cfg.name).lower().strip()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            betas=tuple(float(x) for x in cfg.betas),
            eps=float(cfg.eps),
            amsgrad=bool(cfg.amsgrad),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            betas=tuple(float(x) for x in cfg.betas),
            eps=float(cfg.eps),
            amsgrad=bool(cfg.amsgrad),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            momentum=float(cfg.momentum),
            nesterov=bool(cfg.nesterov),
        )
    raise ValueError(f"Unsupported optimizer name '{cfg.name}'")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
):
    total_steps = max(1, int(total_steps))
    warmup_ratio = float(min(max(warmup_ratio, 0.0), 0.999))
    warmup_steps = int(max(1, round(total_steps * warmup_ratio)))
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    def lr_lambda(step: int):
        s = int(step)
        if s < warmup_steps:
            return float(s + 1) / float(max(1, warmup_steps))
        prog = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = min(max(prog, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))


class MemoryOptimizer:
    """Profiles memory stack performance (latency/energy-ish/accuracy proxy)."""

    def __init__(self, cortex):
        self.cortex = cortex
        self.metrics = {"access_time": [], "energy_usage": [], "mse_proxy": []}

    def reset(self):
        self.metrics = {"access_time": [], "energy_usage": [], "mse_proxy": []}

    @torch.no_grad()
    def profile(self, test_batches):
        self.cortex.eval()
        for x, ctx in test_batches:
            start = time.time()
            out = self.cortex(x, ctx, operation="retrieve")
            dt = time.time() - start
            energy = sum(p.numel() for p in self.cortex.parameters()) / 1e6
            target = ctx
            mse = F.mse_loss(out, target[: out.size(0)], reduction="mean").item()
            self.metrics["access_time"].append(dt)
            self.metrics["energy_usage"].append(energy)
            self.metrics["mse_proxy"].append(mse)
        return {k: float(sum(v) / max(1, len(v))) for k, v in self.metrics.items()}

    def enable_energy_mode_if_slow(self, max_time: float = 0.1):
        avg = float(sum(self.metrics["access_time"]) / max(1, len(self.metrics["access_time"])))
        if avg > max_time:
            self.cortex.enable_energy_mode(True)
            return True
        return False
