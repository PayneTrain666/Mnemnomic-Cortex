from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ContextStabilityReport:
    ok: bool
    finite: bool
    context_norm: float
    mount_delta_norm: float
    max_allowed_norm: float
    max_allowed_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "finite": self.finite,
            "context_norm": self.context_norm,
            "mount_delta_norm": self.mount_delta_norm,
            "max_allowed_norm": self.max_allowed_norm,
            "max_allowed_delta": self.max_allowed_delta,
        }


class ContextStabilityGuard:
    """Runtime stability checks for context mounting."""

    def check(self, context: torch.Tensor, mount_delta: torch.Tensor, rules: Dict[str, float]) -> ContextStabilityReport:
        finite = bool(torch.isfinite(context).all().item() and torch.isfinite(mount_delta).all().item())
        context_norm = float(context.detach().norm().cpu())
        delta_norm = float(mount_delta.detach().norm().cpu())
        max_norm = float(rules.get("max_context_norm", 10.0))
        max_delta = float(rules.get("max_mount_delta", 2.0))
        ok = finite and context_norm <= max_norm * max(1.0, context.numel() ** 0.5 / 100.0) and delta_norm <= max_delta * max(1.0, mount_delta.numel() ** 0.5 / 100.0)
        return ContextStabilityReport(ok, finite, context_norm, delta_norm, max_norm, max_delta)

    def repair(self, mount_delta: torch.Tensor, max_norm: float = 10.0) -> torch.Tensor:
        mount_delta = torch.nan_to_num(mount_delta, nan=0.0, posinf=max_norm, neginf=-max_norm)
        norm = mount_delta.norm().clamp_min(1e-8)
        if norm > max_norm:
            mount_delta = mount_delta * (max_norm / norm)
        return mount_delta
