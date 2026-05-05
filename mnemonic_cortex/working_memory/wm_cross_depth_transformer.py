from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ._wm_light_transformer import LightweightTransformerStack


@dataclass
class WMCrossDepthTransformerConfig:
    """Configuration for cross-depth transformer cognition.

    Contract:
    - input:  [B,Z,T,3,D]
    - output: [B,Z,T,3,D]

    Processing rule:
    - Treat each batch/time/triplet group as a depth sequence.
    - Transformer operates along Z only.
    """

    dim: int
    num_depths: int = 8
    triplet_dim: int = 3
    num_heads: int = 4
    num_layers: int = 1
    ff_multiplier: int = 4
    dropout: float = 0.0
    residual_mix: float = 0.40
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
class WMCrossDepthTransformerTrace:
    input_shape: list
    output_shape: list
    depth_sequence_count: int
    depth_count: int
    num_layers: int
    finite: bool
    cross_depth_delta_norm: float
    depth_energy: list
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WMCrossDepthTransformer(nn.Module):
    """Transformer layer that exchanges information across depth slices."""

    def __init__(self, config: WMCrossDepthTransformerConfig):
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
        nn.init.normal_(self.depth_embedding, std=0.01)
        self.output_norm = nn.LayerNorm(config.dim)
        self.last_trace: Optional[WMCrossDepthTransformerTrace] = None

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

        enriched = x + self.depth_embedding.view(1, z, 1, 1, d)

        # [B,Z,T,3,D] -> [B,T,3,Z,D] -> [B*T*3,Z,D]
        depth_sequences = enriched.permute(0, 2, 3, 1, 4).reshape(b * t * three, z, d)
        processed = self.encoder(depth_sequences)
        processed = self.output_norm(processed)

        out_sequences = (1.0 - self.config.residual_mix) * depth_sequences + self.config.residual_mix * processed
        out = out_sequences.reshape(b, t, three, z, d).permute(0, 3, 1, 2, 4).contiguous()

        finite = bool(torch.isfinite(out).all().item())
        depth_energy = out.detach().pow(2).mean(dim=(0, 2, 3, 4)).cpu().tolist()

        trace = WMCrossDepthTransformerTrace(
            input_shape=list(x.shape),
            output_shape=list(out.shape),
            depth_sequence_count=b * t * three,
            depth_count=z,
            num_layers=self.config.num_layers,
            finite=finite,
            cross_depth_delta_norm=float((out - x).detach().norm().cpu()),
            depth_energy=depth_energy,
            paamax_metadata={
                "trace_type": "wm_cross_depth_transformer",
                "bounded_scope": "across_depth_slices",
                "confidence": 1.0 if finite else 0.0,
                "depth_energy_proxy": depth_energy,
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
