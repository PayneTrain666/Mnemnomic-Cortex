from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class WMCounterfactualAttentionConfig:
    dim: int
    probe_strength: float = 0.20
    improvement_threshold: float = 0.02

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not 0.0 <= self.probe_strength <= 1.0:
            raise ValueError("probe_strength must be in [0,1]")


@dataclass
class WMCounterfactualAttentionOutput:
    output: torch.Tensor
    counterfactual_delta: torch.Tensor
    harmful_memory_score: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "counterfactual_delta_shape": list(self.counterfactual_delta.shape),
            "harmful_memory_score_shape": list(self.harmful_memory_score.shape),
            "trace": self.trace,
        }


class WMCounterfactualAttention(nn.Module):
    """Counterfactual probe for memory context.

    It estimates what changes if memory context is ablated. In WM-3B this emits
    trace metadata only; hard do-not-reuse updates are deferred to later memory
    governance/write stages.
    """

    def __init__(self, config: WMCounterfactualAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.delta_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.Tanh())
        self.last_output: Optional[WMCounterfactualAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        if memory_context is None:
            memory_context = torch.zeros(tokens.size(0), self.config.dim, device=tokens.device, dtype=tokens.dtype)
        if memory_context.dim() != 2 or memory_context.shape != (tokens.size(0), self.config.dim):
            raise ValueError(f"memory_context must be [B,{self.config.dim}]")

        delta = self.delta_proj(memory_context)
        harmful_score = torch.sigmoid(delta.norm(dim=-1) - tokens.mean(dim=1).norm(dim=-1))
        # If harmful score is high, subtract a small portion of memory signal.
        correction = -self.config.probe_strength * harmful_score.view(-1, 1, 1) * delta.unsqueeze(1)
        output = tokens + correction
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_counterfactual_attention",
            "harmful_memory_score": harmful_score.detach().cpu().tolist(),
            "counterfactual_delta_norm": float(delta.detach().norm().cpu()),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_counterfactual_attention",
                "counterfactual_probe": True,
                "do_not_reuse_candidate": bool((harmful_score > 0.75).any().item()),
                "confidence": float((1.0 - harmful_score.mean()).detach().clamp(0, 1).cpu()) if finite else 0.0,
            },
        }
        out = WMCounterfactualAttentionOutput(output=output, counterfactual_delta=delta, harmful_memory_score=harmful_score, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output
