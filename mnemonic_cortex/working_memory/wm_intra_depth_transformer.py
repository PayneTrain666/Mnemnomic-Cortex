"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm intra depth transformer.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_depth_guards import ensure_depth_state, ensure_token_state, ensure_triplet_axis, normalize_quaternion, ensure_quaternion_pack, depth_contract_trace, assert_depth_compatible_tokens

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ._wm_light_transformer import LightweightTransformerStack


@dataclass
class WMIntraDepthTransformerConfig:
    """Configuration for intra-depth transformer processing.

    Contract:
    - input:  [B,Z,T,3,D]
    - output: [B,Z,T,3,D]

    Processing rule:
    - Treat every depth/triplet pair as a separate temporal stream.
    - Transformer operates along T only.
    """

    dim: int
    num_depths: int = 8
    triplet_dim: int = 3
    num_heads: int = 4
    num_layers: int = 1
    ff_multiplier: int = 4
    dropout: float = 0.0
    residual_mix: float = 0.50
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMIntraDepthTransformerTrace:
    input_shape: list
    output_shape: list
    stream_count: int
    sequence_length: int
    num_layers: int
    finite: bool
    temporal_delta_norm: float
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WMIntraDepthTransformer(nn.Module):
    """Transformer layer for temporal cognition inside each depth/triplet stream."""

    def __init__(self, config: WMIntraDepthTransformerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = LightweightTransformerStack(
            dim=config.dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_multiplier=config.ff_multiplier,
            dropout=config.dropout,
        )
        self.depth_embedding = nn.Parameter(torch.zeros(config.num_depths, config.dim))
        self.triplet_embedding = nn.Parameter(torch.zeros(config.triplet_dim, config.dim))
        nn.init.normal_(self.depth_embedding, std=0.01)
        nn.init.normal_(self.triplet_embedding, std=0.01)
        self.output_norm = nn.LayerNorm(config.dim)
        self.last_trace: Optional[WMIntraDepthTransformerTrace] = None

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,Z,T,3,D], got rank {x.dim()}")
        b, z, t, three, d = x.shape
        if z != self.config.num_depths:
            raise ValueError(f"Expected Z={self.config.num_depths}, got {z}")
        if three != self.config.triplet_dim:
            raise ValueError(f"Expected triplet_dim={self.config.triplet_dim}, got {three}")
        if d != self.config.dim:
            raise ValueError(f"Expected D={self.config.dim}, got {d}")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf")

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        self._validate_input(x)
        b, z, t, three, d = x.shape

        depth_bias = self.depth_embedding.view(1, z, 1, 1, d)
        triplet_bias = self.triplet_embedding.view(1, 1, 1, three, d)
        enriched = x + depth_bias + triplet_bias

        # [B,Z,T,3,D] -> [B,Z,3,T,D] -> [B*Z*3,T,D]
        streams = enriched.permute(0, 1, 3, 2, 4).reshape(b * z * three, t, d)
        processed = self.encoder(streams)
        processed = self.output_norm(processed)

        out_streams = (1.0 - self.config.residual_mix) * streams + self.config.residual_mix * processed
        out = out_streams.reshape(b, z, three, t, d).permute(0, 1, 3, 2, 4).contiguous()

        finite = bool(torch.isfinite(out).all().item())
        trace = WMIntraDepthTransformerTrace(
            input_shape=list(x.shape),
            output_shape=list(out.shape),
            stream_count=b * z * three,
            sequence_length=t,
            num_layers=self.config.num_layers,
            finite=finite,
            temporal_delta_norm=float((out - x).detach().norm().cpu()),
            paamax_metadata={
                "trace_type": "wm_intra_depth_transformer",
                "bounded_scope": "within_depth_triplet_temporal_stream",
                "confidence": 1.0 if finite else 0.0,
            },
        )
        self.last_trace = trace

        if return_trace:
            return out, trace.to_dict()
        return out

    def stability_report(self, x: torch.Tensor) -> Dict[str, Any]:
        out, trace = self.forward(x, return_trace=True)
        return {
            "ok": bool(trace["finite"] and list(out.shape) == list(x.shape)),
            "finite": trace["finite"],
            "shape_ok": list(out.shape) == list(x.shape),
            "trace": trace,
        }


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
