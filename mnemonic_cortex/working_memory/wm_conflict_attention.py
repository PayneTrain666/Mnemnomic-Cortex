from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WMConflictAttentionConfig:
    dim: int
    conflict_threshold: float = 0.65
    residual_mix: float = 0.10
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not 0.0 <= self.conflict_threshold <= 1.0:
            raise ValueError("conflict_threshold must be in [0,1]")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMConflictAttentionOutput:
    output: torch.Tensor
    conflict_score: torch.Tensor
    quarantine_mask: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "conflict_score_shape": list(self.conflict_score.shape),
            "quarantine_mask": self.quarantine_mask.detach().cpu().tolist(),
            "trace": self.trace,
        }


class WMConflictAttention(nn.Module):
    """Conflict-aware attention with quarantine metadata."""

    def __init__(self, config: WMConflictAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.repair_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMConflictAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        summary = tokens.mean(dim=1)
        if memory_context is None:
            memory_context = summary
        if memory_context.dim() != 2 or memory_context.shape != (tokens.size(0), self.config.dim):
            raise ValueError(f"memory_context must be [B,{self.config.dim}]")

        sim = F.cosine_similarity(summary, memory_context, dim=-1, eps=self.config.eps)
        conflict_score = torch.clamp((1.0 - sim) / 2.0, 0.0, 1.0)
        quarantine_mask = conflict_score > self.config.conflict_threshold
        repair = self.repair_proj(summary - memory_context).unsqueeze(1)
        output = tokens - self.config.residual_mix * conflict_score.view(-1, 1, 1) * repair
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_conflict_attention",
            "conflict_score": conflict_score.detach().cpu().tolist(),
            "quarantine_mask": quarantine_mask.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_conflict_attention",
                "conflict_quarantine": bool(quarantine_mask.any().item()),
                "quarantine_required": bool(quarantine_mask.any().item()),
                "confidence": float((1.0 - conflict_score.mean()).detach().cpu()) if finite else 0.0,
            },
        }
        out = WMConflictAttentionOutput(output=output, conflict_score=conflict_score, quarantine_mask=quarantine_mask, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output
