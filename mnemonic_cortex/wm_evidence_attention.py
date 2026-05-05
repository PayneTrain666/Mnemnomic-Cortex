from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WMEvidenceAttentionConfig:
    """Evidence-structured attention configuration.

    Contract:
    - tokens: [B,T,D]
    - memory_context: optional [B,D]
    - output: [B,T,D]
    """

    dim: int
    evidence_slots: int = 4
    residual_mix: float = 0.10
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.evidence_slots <= 0:
            raise ValueError("evidence_slots must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMEvidenceAttentionOutput:
    output: torch.Tensor
    evidence_context: torch.Tensor
    evidence_scores: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "evidence_context_shape": list(self.evidence_context.shape),
            "evidence_scores_shape": list(self.evidence_scores.shape),
            "trace": self.trace,
        }


class WMEvidenceAttention(nn.Module):
    """Evidence-structured attention over current WM tokens.

    This is not a document retriever. It creates a compact internal evidence
    summary from the current token state and optional memory context, then emits
    auditable evidence scores for downstream conflict/counterfactual modules.
    """

    def __init__(self, config: WMEvidenceAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.query = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.evidence_bank = nn.Parameter(torch.randn(config.evidence_slots, config.dim) * 0.02)
        self.context_gate = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.Sigmoid())
        self.out_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMEvidenceAttentionOutput] = None

    def _validate(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor]) -> None:
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        if memory_context is not None:
            if memory_context.dim() != 2 or memory_context.shape != (tokens.size(0), self.config.dim):
                raise ValueError(f"memory_context must be [B,{self.config.dim}], got {tuple(memory_context.shape)}")
            if not torch.isfinite(memory_context).all():
                raise ValueError("memory_context contains NaN or Inf")

    def forward(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor] = None, return_trace: bool = False):
        self._validate(tokens, memory_context)
        summary = tokens.mean(dim=1)
        q = F.normalize(self.query(summary), dim=-1, eps=self.config.eps)
        bank = F.normalize(self.evidence_bank, dim=-1, eps=self.config.eps)
        scores = torch.matmul(q, bank.t())
        weights = torch.softmax(scores, dim=-1)
        evidence_context = torch.matmul(weights, self.evidence_bank)

        if memory_context is not None:
            gate = self.context_gate(memory_context)
            evidence_context = 0.65 * evidence_context + 0.35 * gate * memory_context

        delta = self.out_proj(evidence_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta
        finite = bool(torch.isfinite(output).all().item())
        confidence = float((1.0 / (1.0 + scores.var(dim=-1).mean())).detach().cpu())
        trace = {
            "trace_type": "wm_evidence_attention",
            "evidence_slots": self.config.evidence_slots,
            "evidence_scores": scores.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_evidence_attention",
                "confidence": confidence if finite else 0.0,
                "audit_metadata": {"evidence_slots": self.config.evidence_slots},
            },
        }
        out = WMEvidenceAttentionOutput(output=output, evidence_context=evidence_context, evidence_scores=scores, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output

    def stability_report(self, tokens: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out, trace = self.forward(tokens, memory_context=memory_context, return_trace=True)
        return {"ok": bool(torch.isfinite(out).all().item()), "trace": trace}
