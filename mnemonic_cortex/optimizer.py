"""
Plain-language summary
----------------------
What this file is for: Builds optimizers and learning-rate schedules for training.
How it fits in the system: Training support, not a memory system.
Status: WORKING
Important notes for non-coders: Used by smoke training and some tool scripts.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

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


def unique_trainable_parameters(params: Iterable) -> list:
    """Return trainable parameters once each, preserving their first-seen order."""
    unique = []
    seen_ids = set()
    for param in params:
        if not getattr(param, "requires_grad", False):
            continue
        param_id = id(param)
        if param_id in seen_ids:
            continue
        seen_ids.add(param_id)
        unique.append(param)
    return unique


def _option_values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, int)) else False


def _group_options_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        _option_values_equal(left[key], right[key]) for key in left
    )


def _prepare_parameter_groups(
    groups: Iterable[Mapping[str, Any]],
    defaults: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prepared = []
    seen: dict[int, tuple[int, dict[str, Any]]] = {}

    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise TypeError(
                "optimizer parameters must be all Parameters or all param-group dictionaries"
            )
        if "params" not in group:
            raise ValueError(f"optimizer param group {group_index} is missing 'params'")

        raw_params = group["params"]
        if isinstance(raw_params, torch.Tensor):
            raw_params = [raw_params]

        options = {key: value for key, value in group.items() if key != "params"}
        effective_options = dict(defaults)
        effective_options.update(options)
        unique_params = []

        for param in raw_params:
            if not getattr(param, "requires_grad", False):
                continue
            param_id = id(param)
            previous = seen.get(param_id)
            if previous is None:
                seen[param_id] = (group_index, effective_options)
                unique_params.append(param)
                continue

            previous_index, previous_options = previous
            if not _group_options_equal(previous_options, effective_options):
                raise ValueError(
                    "the same Parameter appears in optimizer param groups "
                    f"{previous_index} and {group_index} with conflicting options"
                )
            # An identity duplicate with identical effective options is harmless.

        if unique_params:
            prepared.append({"params": unique_params, **options})

    return prepared


def build_optimizer(params: Iterable, cfg: OptimizerConfig) -> Optimizer:
    supplied = list(params)
    name = str(cfg.name).strip().lower()
    defaults = {
        "lr": float(cfg.lr),
        "betas": tuple(cfg.betas),
        "eps": float(cfg.eps),
    }
    if name == "adamw":
        defaults["weight_decay"] = float(cfg.weight_decay)
    elif name != "adam":
        raise ValueError(f"unknown optimizer name={cfg.name}")

    contains_groups = bool(supplied) and isinstance(supplied[0], Mapping)
    if contains_groups:
        trainable = _prepare_parameter_groups(supplied, defaults)
    else:
        if any(isinstance(item, Mapping) for item in supplied):
            raise TypeError(
                "optimizer parameters must be all Parameters or all param-group dictionaries"
            )
        trainable = unique_trainable_parameters(supplied)

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


def optimizer_references_parameters(
    optimizer: Optimizer,
    parameters: Iterable,
) -> bool:
    """Return whether an optimizer holds any listed Parameter by identity."""
    parameter_ids = {id(param) for param in parameters}
    return any(
        id(param) in parameter_ids
        for group in optimizer.param_groups
        for param in group["params"]
    )


def guard_optimizer_parameter_replacement(
    optimizer: Optimizer,
    parameters: Iterable,
) -> None:
    """Refuse structural replacement while the optimizer still holds old params."""
    if optimizer_references_parameters(optimizer, parameters):
        raise RuntimeError(
            "unsafe Parameter replacement: the optimizer still references a Parameter "
            "slated for structural replacement; rebuild the optimizer after replacement "
            "so parameter groups and optimizer state cannot become stale"
        )


def migrate_optimizer_parameter_replacements(
    optimizer: Optimizer,
    replacements: Mapping | Iterable[tuple],
) -> Optimizer:
    """Guard an unsupported in-place migration and require a safe optimizer rebuild."""
    pairs = replacements.items() if isinstance(replacements, Mapping) else replacements
    old_parameters = [old_param for old_param, _new_param in pairs]
    guard_optimizer_parameter_replacement(optimizer, old_parameters)
    return optimizer


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
