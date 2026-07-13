"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm triplet state.
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
class WMTripletStateConfig:
    dim: int
    triplet_dim: int = 3
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")


@dataclass
class WMTripletState:
    """Explicit anchor/direction/phase state container."""

    anchor: torch.Tensor
    direction: torch.Tensor
    phase: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tensor(self) -> torch.Tensor:
        return torch.stack([self.anchor, self.direction, self.phase], dim=-2)

    def shape_summary(self) -> Dict[str, Any]:
        return {
            "anchor": list(self.anchor.shape),
            "direction": list(self.direction.shape),
            "phase": list(self.phase.shape),
            "tensor": list(self.tensor.shape),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_summary": self.shape_summary(),
            "metadata": self.metadata,
        }


class WMTripletProjector(nn.Module):
    """Project token states into anchor/direction/phase triplets.

    Input:
    - x [B,T,D]

    Output:
    - WMTripletState with each component [B,T,D]
    - tensor view [B,T,3,D]
    """

    def __init__(self, dim: int):
        super().__init__()
        cfg = WMTripletStateConfig(dim=dim)
        cfg.validate()
        self.dim = dim
        self.proj = nn.Linear(dim, 3 * dim)
        self.fuse_proj = nn.Linear(3 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def project(self, x: torch.Tensor) -> WMTripletState:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"Expected x [B,T,{self.dim}], got {tuple(x.shape)}")
        raw = self.proj(self.norm(x))
        anchor, direction, phase = raw.chunk(3, dim=-1)
        state = WMTripletState(
            anchor=anchor,
            direction=direction,
            phase=torch.sin(phase),
            metadata={
                "trace_type": "wm_triplet_state",
                "triplet_axis": ["anchor", "direction", "phase"],
            },
        )
        return state

    def fuse(self, state: WMTripletState) -> torch.Tensor:
        tensor = state.tensor
        if tensor.size(-2) != 3:
            raise ValueError("triplet state must have triplet dimension 3")
        return self.fuse_proj(tensor.reshape(*tensor.shape[:-2], 3 * self.dim))

    def forward(self, x: torch.Tensor, return_state: bool = False):
        state = self.project(x)
        fused = self.fuse(state)
        if return_state:
            return fused, state
        return fused


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
