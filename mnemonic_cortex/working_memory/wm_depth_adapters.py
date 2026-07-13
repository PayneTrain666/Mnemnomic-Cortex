"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm depth adapters.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_depth_guards import ensure_depth_state, ensure_token_state, ensure_triplet_axis, normalize_quaternion, ensure_quaternion_pack, depth_contract_trace, assert_depth_compatible_tokens

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class WMDepthAdaptersConfig:
    dim: int
    num_depths: int = 8
    triplet_dim: int = 3
    adapter_hidden_multiplier: int = 2
    residual_mix: float = 0.35
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMDepthAdaptersTrace:
    input_shape: list
    output_shape: list
    finite: bool
    adapter_delta_norm: float
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WMDepthAdapters(nn.Module):
    """Per-depth/per-triplet lightweight adapters for [B,Z,T,3,D]."""

    def __init__(self, config: WMDepthAdaptersConfig):
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.dim * config.adapter_hidden_multiplier
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, config.dim),
                )
                for _ in range(config.num_depths * config.triplet_dim)
            ]
        )
        self.last_trace: Optional[WMDepthAdaptersTrace] = None

    def _validate(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,Z,T,3,D], got rank {x.dim()}")
        b, z, t, three, d = x.shape
        if z != self.config.num_depths or three != self.config.triplet_dim or d != self.config.dim:
            raise ValueError(f"Expected [B,{self.config.num_depths},T,{self.config.triplet_dim},{self.config.dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        self._validate(x)
        b, z, t, three, d = x.shape
        outputs = []
        idx = 0
        for zi in range(z):
            triplet_out = []
            for ti in range(three):
                stream = x[:, zi, :, ti, :]
                adapted = self.adapters[idx](stream)
                mixed = (1.0 - self.config.residual_mix) * stream + self.config.residual_mix * adapted
                triplet_out.append(mixed)
                idx += 1
            outputs.append(torch.stack(triplet_out, dim=2))  # [B,T,3,D]
        out = torch.stack(outputs, dim=1)  # [B,Z,T,3,D]
        finite = bool(torch.isfinite(out).all().item())
        trace = WMDepthAdaptersTrace(
            input_shape=list(x.shape),
            output_shape=list(out.shape),
            finite=finite,
            adapter_delta_norm=float((out - x).detach().norm().cpu()),
            paamax_metadata={
                "trace_type": "wm_depth_adapters",
                "confidence": 1.0 if finite else 0.0,
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
