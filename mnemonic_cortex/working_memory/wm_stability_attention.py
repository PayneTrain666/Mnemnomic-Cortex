from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class WMStabilityAttentionConfig:
    dim: int
    max_norm: float = 100.0
    residual_mix: float = 0.05
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.max_norm <= 0:
            raise ValueError("max_norm must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMStabilityAttentionOutput:
    output: torch.Tensor
    stability_score: torch.Tensor
    repaired: bool
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "stability_score": self.stability_score.detach().cpu().tolist(),
            "repaired": self.repaired,
            "trace": self.trace,
        }


class WMStabilityAttention(nn.Module):
    """Stability-aware attention and repair clamp."""

    def __init__(self, config: WMStabilityAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.repair_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.Tanh())
        self.last_output: Optional[WMStabilityAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        finite_before = bool(torch.isfinite(tokens).all().item())
        cleaned = torch.nan_to_num(tokens, nan=0.0, posinf=self.config.max_norm, neginf=-self.config.max_norm)
        norm = cleaned.norm(dim=-1)
        over = norm > self.config.max_norm
        scale = torch.clamp(self.config.max_norm / norm.clamp_min(self.config.eps), max=1.0)
        repaired_tokens = cleaned * scale.unsqueeze(-1)
        repair_delta = self.repair_proj(repaired_tokens.mean(dim=1)).unsqueeze(1)
        output = repaired_tokens + self.config.residual_mix * repair_delta
        finite_after = bool(torch.isfinite(output).all().item())
        mean_norm = output.norm(dim=-1).mean(dim=-1)
        stability_score = 1.0 / (1.0 + mean_norm / self.config.max_norm)
        repaired = bool((not finite_before) or over.any().item())
        trace = {
            "trace_type": "wm_stability_attention",
            "finite_before": finite_before,
            "finite_after": finite_after,
            "repaired": repaired,
            "max_norm": self.config.max_norm,
            "stability_score": stability_score.detach().cpu().tolist(),
            "paamax_metadata": {
                "trace_type": "wm_stability_attention",
                "stability_guard": True,
                "repaired": repaired,
                "confidence": float(stability_score.detach().mean().cpu()) if finite_after else 0.0,
            },
        }
        out = WMStabilityAttentionOutput(output=output, stability_score=stability_score, repaired=repaired, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output
