from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurvatureMetricPolicyConfig:
    """Configuration for global/per-slot/per-depth/context curvature policy."""

    num_slots: int
    num_depths: int = 8
    context_dim: int = 128
    curvature_min: float = -5.0
    curvature_max: float = 5.0
    drift_weight: float = 0.01
    eps: float = 1e-8

    def validate(self) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.curvature_min >= self.curvature_max:
            raise ValueError("curvature_min must be < curvature_max")
        if self.drift_weight < 0:
            raise ValueError("drift_weight must be non-negative")


@dataclass
class CurvatureMetricPolicyOutput:
    global_curvature: torch.Tensor
    per_slot_curvature: torch.Tensor
    per_depth_curvature: torch.Tensor
    context_curvature: torch.Tensor
    combined_curvature: torch.Tensor
    drift_penalty: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_curvature": self.global_curvature.detach().cpu().tolist(),
            "per_slot_curvature": self.per_slot_curvature.detach().cpu().tolist(),
            "per_depth_curvature": self.per_depth_curvature.detach().cpu().tolist(),
            "context_curvature": self.context_curvature.detach().cpu().tolist(),
            "combined_shape": list(self.combined_curvature.shape),
            "drift_penalty": float(self.drift_penalty.detach().cpu()),
            "trace": self.trace,
        }


class CurvatureMetricPolicy(nn.Module):
    """Curvature policy for curved WM slots and depth slices.

    Produces bounded curvature for:
    - global curvature
    - per-slot curvature
    - per-depth curvature
    - context-conditioned curvature

    Output combined curvature shape:
    - [B,Z,S] where B=batch, Z=depth, S=slot.
    """

    def __init__(self, config: CurvatureMetricPolicyConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.global_curvature = nn.Parameter(torch.zeros(1))
        self.per_slot_curvature = nn.Parameter(torch.zeros(config.num_slots))
        self.per_depth_curvature = nn.Parameter(torch.zeros(config.num_depths))
        self.context_to_curvature = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, max(16, config.context_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, config.context_dim // 2), 1),
        )

        self.register_buffer("reference_global", torch.zeros(1))
        self.register_buffer("reference_slot", torch.zeros(config.num_slots))
        self.register_buffer("reference_depth", torch.zeros(config.num_depths))
        self.last_trace: Optional[Dict[str, Any]] = None

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.nan_to_num(x), self.config.curvature_min, self.config.curvature_max)

    def clamped_components(self) -> Dict[str, torch.Tensor]:
        return {
            "global": self._clamp(self.global_curvature),
            "slot": self._clamp(self.per_slot_curvature),
            "depth": self._clamp(self.per_depth_curvature),
        }

    def drift_penalty(self) -> torch.Tensor:
        comps = self.clamped_components()
        penalty = (
            F.mse_loss(comps["global"], self.reference_global)
            + F.mse_loss(comps["slot"], self.reference_slot)
            + F.mse_loss(comps["depth"], self.reference_depth)
        )
        return self.config.drift_weight * penalty

    @torch.no_grad()
    def set_reference_to_current(self) -> None:
        comps = self.clamped_components()
        self.reference_global.copy_(comps["global"])
        self.reference_slot.copy_(comps["slot"])
        self.reference_depth.copy_(comps["depth"])

    def context_conditioned_curvature(self, context: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if context is None:
            return torch.zeros(batch_size, 1, device=self.global_curvature.device, dtype=self.global_curvature.dtype)
        if context.dim() == 3:
            summary = context.mean(dim=1)
        elif context.dim() == 2:
            summary = context
        else:
            raise ValueError("context must be [B,C,D] or [B,D]")
        if summary.size(-1) != self.config.context_dim:
            raise ValueError(f"context dim must be {self.config.context_dim}, got {summary.size(-1)}")
        return self._clamp(self.context_to_curvature(summary))

    def forward(self, context: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> CurvatureMetricPolicyOutput:
        if batch_size is None:
            batch_size = int(context.size(0)) if context is not None else 1

        comps = self.clamped_components()
        context_curv = self.context_conditioned_curvature(context, batch_size)

        # [B,Z,S] = global + context + depth + slot
        combined = (
            comps["global"].view(1, 1, 1)
            + context_curv.view(batch_size, 1, 1)
            + comps["depth"].view(1, self.config.num_depths, 1)
            + comps["slot"].view(1, 1, self.config.num_slots)
        )
        combined = self._clamp(combined)
        penalty = self.drift_penalty()

        trace = {
            "batch_size": batch_size,
            "num_depths": self.config.num_depths,
            "num_slots": self.config.num_slots,
            "curvature_min": self.config.curvature_min,
            "curvature_max": self.config.curvature_max,
            "combined_shape": list(combined.shape),
            "drift_penalty": float(penalty.detach().cpu()),
            "paamax_metadata": {
                "trace_type": "curvature_metric_policy",
                "conflict_check_recommended": bool((combined.abs() > 0.9 * self.config.curvature_max).any().item()),
                "write_permission_required": False,
            },
        }
        self.last_trace = trace
        return CurvatureMetricPolicyOutput(
            global_curvature=comps["global"],
            per_slot_curvature=comps["slot"],
            per_depth_curvature=comps["depth"],
            context_curvature=context_curv,
            combined_curvature=combined,
            drift_penalty=penalty,
            trace=trace,
        )

    @torch.no_grad()
    def repair_in_place(self) -> Dict[str, Any]:
        comps = self.clamped_components()
        self.global_curvature.data.copy_(comps["global"])
        self.per_slot_curvature.data.copy_(comps["slot"])
        self.per_depth_curvature.data.copy_(comps["depth"])
        return self.validate_policy()

    def validate_policy(self) -> Dict[str, Any]:
        comps = self.clamped_components()
        finite = all(torch.isfinite(v).all().item() for v in comps.values())
        bounded = all(((v >= self.config.curvature_min) & (v <= self.config.curvature_max)).all().item() for v in comps.values())
        return {
            "finite": bool(finite),
            "bounded": bool(bounded),
            "ok": bool(finite and bounded),
            "global_shape": list(comps["global"].shape),
            "slot_shape": list(comps["slot"].shape),
            "depth_shape": list(comps["depth"].shape),
        }


# ---------------------------------------------------------------------------
# WM-QD-1A foundation-quality contract
# ---------------------------------------------------------------------------

def wm_qd1a_foundation_contract() -> dict:
    """Return serialization-safe quality metadata for this early-WM module.

    This does not mutate runtime state. It exists so the quality tooling can
    verify that the module has an explicit contract for shape/finite checks,
    traceability, PAAMA-X metadata, fallback behavior, and boundedness.
    """
    return foundation_trace(
        trace_type="wm_qd1a_foundation_contract",
        module=__name__,
        message="early working-memory foundation module hardened by WM-QD-1A",
        payload={
            "shape_checks_required": True,
            "finite_checks_required": True,
            "serialization_safe": True,
            "trace_hooks_required": True,
            "paamax_metadata_required": True,
            "boundedness_required": True,
            "runtime_mutation": "no automatic mutation by quality tooling",
        },
    )
