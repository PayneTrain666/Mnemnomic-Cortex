from __future__ import annotations

from .wm_depth_guards import ensure_depth_state, ensure_token_state, ensure_triplet_axis, normalize_quaternion, ensure_quaternion_pack, depth_contract_trace, assert_depth_compatible_tokens

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class WMDepthFusionConfig:
    dim: int
    num_depths: int = 8
    triplet_dim: int = 3
    residual_weight: float = 0.50
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")
        if not 0.0 <= self.residual_weight <= 1.0:
            raise ValueError("residual_weight must be in [0,1]")


@dataclass
class WMDepthFusionTrace:
    input_shape: list
    output_shape: list
    depth_weights: list
    triplet_weights: list
    finite: bool
    disagreement: float
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WMDepthFusion(nn.Module):
    """Fuse [B,Z,T,3,D] depth/triplet state back to [B,T,D]."""

    def __init__(self, config: WMDepthFusionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.depth_logits = nn.Parameter(torch.zeros(config.num_depths))
        self.triplet_logits = nn.Parameter(torch.tensor([0.35, 0.20, 0.10], dtype=torch.float32))
        self.output_norm = nn.LayerNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.dim)
        self.last_trace: Optional[WMDepthFusionTrace] = None

    def _validate(self, depth_state: torch.Tensor) -> None:
        if depth_state.dim() != 5:
            raise ValueError(f"Expected depth_state [B,Z,T,3,D], got rank {depth_state.dim()}")
        b, z, t, three, d = depth_state.shape
        if z != self.config.num_depths or three != self.config.triplet_dim or d != self.config.dim:
            raise ValueError(f"Expected [B,{self.config.num_depths},T,{self.config.triplet_dim},{self.config.dim}], got {tuple(depth_state.shape)}")
        if not torch.isfinite(depth_state).all():
            raise ValueError("depth_state contains NaN or Inf")

    def forward(self, depth_state: torch.Tensor, residual: Optional[torch.Tensor] = None, return_trace: bool = False):
        self._validate(depth_state)
        b, z, t, three, d = depth_state.shape
        if residual is not None and (residual.dim() != 3 or residual.shape != (b, t, d)):
            raise ValueError(f"residual must be [B,T,D] matching depth_state, got {None if residual is None else tuple(residual.shape)}")

        depth_weights = torch.softmax(self.depth_logits, dim=0)
        triplet_weights = torch.softmax(self.triplet_logits, dim=0)
        weighted = depth_state * depth_weights.view(1, z, 1, 1, 1) * triplet_weights.view(1, 1, 1, three, 1)
        fused = weighted.sum(dim=(1, 3))
        fused = self.output_proj(self.output_norm(fused))

        if residual is not None:
            out = (1.0 - self.config.residual_weight) * fused + self.config.residual_weight * residual
        else:
            out = fused

        finite = bool(torch.isfinite(out).all().item())
        disagreement = float(depth_state.detach().var(dim=1).mean().cpu())
        trace = WMDepthFusionTrace(
            input_shape=list(depth_state.shape),
            output_shape=list(out.shape),
            depth_weights=depth_weights.detach().cpu().tolist(),
            triplet_weights=triplet_weights.detach().cpu().tolist(),
            finite=finite,
            disagreement=disagreement,
            paamax_metadata={
                "trace_type": "wm_depth_fusion",
                "confidence": float(1.0 / (1.0 + disagreement)) if finite else 0.0,
                "disagreement": disagreement,
            },
        )
        self.last_trace = trace
        if return_trace:
            return out, trace.to_dict()
        return out


# ---------------------------------------------------------------------------
# WM-QD-2A quaternion-depth quality contract
# ---------------------------------------------------------------------------

def wm_qd2a_depth_contract() -> dict:
    """Return serialization-safe quality metadata for this depth/assembly module.

    This is a no-mutation contract used by the quality-deepening tooling. It
    declares the expected [B,Z,T,3,D] depth-state invariants, [B,T,D] token-state
    compatibility, quaternion normalization requirement, trace serialization,
    PAAMA-X metadata, fallback behavior, and boundedness expectations.
    """
    return depth_contract_trace(module=__name__)
