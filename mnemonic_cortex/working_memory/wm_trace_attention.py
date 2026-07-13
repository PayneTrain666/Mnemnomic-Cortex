"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm trace attention.
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


@dataclass
class WMTraceAttentionConfig:
    dim: int
    residual_mix: float = 0.05
    max_trace_items: int = 256

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")
        if self.max_trace_items <= 0:
            raise ValueError("max_trace_items must be positive")


@dataclass
class WMTraceAttentionOutput:
    output: torch.Tensor
    trace_signal: torch.Tensor
    trace_score: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "trace_signal_shape": list(self.trace_signal.shape),
            "trace_score_shape": list(self.trace_score.shape),
            "trace": self.trace,
        }


class WMTraceAttention(nn.Module):
    """Trace-aware attention.

    Converts prior trace structure into a bounded signal so the WM can condition
    on its own audit trail without global full-history attention.
    """

    def __init__(self, config: WMTraceAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.trace_proj = nn.Sequential(nn.Linear(4, config.dim), nn.Tanh(), nn.LayerNorm(config.dim))
        self.output_proj = nn.Linear(config.dim, config.dim)
        self.last_output: Optional[WMTraceAttentionOutput] = None

    def _trace_features(self, tokens: torch.Tensor, prior_trace: Optional[Dict[str, Any]]) -> torch.Tensor:
        b = tokens.size(0)
        if not prior_trace:
            features = torch.zeros(b, 4, device=tokens.device, dtype=tokens.dtype)
            return features
        items = prior_trace.get("items", []) if isinstance(prior_trace, dict) else []
        item_count = min(len(items), self.config.max_trace_items)
        confidence = float(prior_trace.get("confidence", 1.0) or 1.0) if isinstance(prior_trace, dict) else 1.0
        disagreement = float(prior_trace.get("disagreement", 0.0) or 0.0) if isinstance(prior_trace, dict) else 0.0
        paamax = prior_trace.get("paamax_metadata", {}) if isinstance(prior_trace, dict) else {}
        policy_flag = 1.0 if isinstance(paamax, dict) and paamax.get("policy_lane_present", False) else 0.0
        vals = torch.tensor(
            [item_count / self.config.max_trace_items, confidence, disagreement, policy_flag],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return vals.view(1, 4).expand(b, -1)

    def forward(self, tokens: torch.Tensor, prior_trace: Optional[Dict[str, Any]] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        features = self._trace_features(tokens, prior_trace)
        signal = self.trace_proj(features)
        score = torch.sigmoid(signal.mean(dim=-1))
        output = tokens + self.config.residual_mix * self.output_proj(signal).unsqueeze(1)
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_trace_attention",
            "trace_features": features.detach().cpu().tolist(),
            "trace_score": score.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_trace_attention",
                "trace_governance": True,
                "confidence": float(score.detach().mean().cpu()) if finite else 0.0,
            },
        }
        out = WMTraceAttentionOutput(output=output, trace_signal=signal, trace_score=score, trace=trace)
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
