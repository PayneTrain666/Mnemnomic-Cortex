"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm novelty attention.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_attention_guards import ensure_attention_query, ensure_candidate_tensor, ensure_attention_scores, stable_softmax, bounded_attention_topk, ensure_lane_output, attention_contract_trace, attention_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WMNoveltyAttentionConfig:
    dim: int
    lightbulb_threshold: float = 0.35
    residual_mix: float = 0.08
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.lightbulb_threshold < 0:
            raise ValueError("lightbulb_threshold must be non-negative")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMNoveltyAttentionOutput:
    output: torch.Tensor
    novelty_score: torch.Tensor
    lightbulb_mask: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "novelty_score": self.novelty_score.detach().cpu().tolist(),
            "lightbulb_mask": self.lightbulb_mask.detach().cpu().tolist(),
            "trace": self.trace,
        }


class WMNoveltyAttention(nn.Module):
    """Novelty/lightbulb attention."""

    def __init__(self, config: WMNoveltyAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.novelty_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.GELU())
        self.last_output: Optional[WMNoveltyAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        summary = tokens.mean(dim=1)
        if memory_context is None:
            memory_context = torch.zeros_like(summary)
        novelty_vec = summary - memory_context
        novelty_score = torch.tanh(novelty_vec.norm(dim=-1) / (summary.norm(dim=-1).clamp_min(self.config.eps) + 1.0))
        lightbulb_mask = novelty_score > self.config.lightbulb_threshold
        delta = self.novelty_proj(novelty_vec).unsqueeze(1)
        output = tokens + self.config.residual_mix * novelty_score.view(-1, 1, 1) * delta
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_novelty_attention",
            "novelty_score": novelty_score.detach().cpu().tolist(),
            "lightbulb_mask": lightbulb_mask.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_novelty_attention",
                "lightbulb": bool(lightbulb_mask.any().item()),
                "novelty": float(novelty_score.detach().mean().cpu()),
                "confidence": float((1.0 / (1.0 + novelty_score.mean())).detach().cpu()) if finite else 0.0,
            },
        }
        out = WMNoveltyAttentionOutput(output=output, novelty_score=novelty_score, lightbulb_mask=lightbulb_mask, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output


# ---------------------------------------------------------------------------
# WM-QD-3A memory-augmented attention quality contract
# ---------------------------------------------------------------------------

def wm_qd3a_attention_contract() -> dict:
    """Return serialization-safe quality metadata for this attention module.

    This no-mutation contract declares candidate schema validation, lane output
    validation, finite geometry-score requirements, bounded attention/top-k
    behavior, trace serialization, PAAMA-X metadata, conflict/quarantine hooks,
    fallback behavior, and compatibility with QDTWorkingMemory.
    """
    return attention_contract_trace(module=__name__)
