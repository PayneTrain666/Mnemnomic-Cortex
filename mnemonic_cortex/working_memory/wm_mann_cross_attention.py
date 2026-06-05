from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wm_external_memory_interfaces import ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank


@dataclass
class WMMANNCrossAttentionConfig:
    dim: int
    top_k: int = 4
    residual_mix: float = 0.20
    hops: int = 3
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.hops <= 0:
            raise ValueError("hops must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMMANNTraceVisibility:
    pre_fusion_output: torch.Tensor
    per_hop_attention: torch.Tensor
    scratchpad_tokens: torch.Tensor
    confidence: torch.Tensor
    disagreement: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pre_fusion_output_shape": list(self.pre_fusion_output.shape),
            "per_hop_attention_shape": list(self.per_hop_attention.shape),
            "scratchpad_tokens_shape": list(self.scratchpad_tokens.shape),
            "confidence": self.confidence.detach().cpu().tolist(),
            "disagreement": self.disagreement.detach().cpu().tolist(),
        }


@dataclass
class WMMANNCrossAttentionOutput:
    output: torch.Tensor
    memory_context: torch.Tensor
    response: ExternalMemoryResponse
    visibility: WMMANNTraceVisibility
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "memory_context_shape": list(self.memory_context.shape),
            "response": self.response.to_dict(),
            "visibility": self.visibility.to_dict(),
            "trace": self.trace,
        }


class WMMANNCrossAttention(nn.Module):
    """MANN cross-attention with required trace visibility."""

    def __init__(self, config: WMMANNCrossAttentionConfig, external_bank: Optional[SyntheticExternalMemoryBank] = None):
        super().__init__()
        config.validate()
        self.config = config
        self.external_bank = external_bank or SyntheticExternalMemoryBank("mann", config.dim, slots=max(config.top_k, 8), hops=config.hops)
        self.query_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.context_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMMANNCrossAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, depth_state: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        query_state = self.query_proj(tokens.mean(dim=1))
        request = ExternalMemoryQuery("mann", query_state=query_state, depth_state=depth_state, context=context, metadata={"source": "WMMANNCrossAttention"})
        response = self.external_bank.query(request, top_k=self.config.top_k)
        weights = torch.softmax(response.scores, dim=-1)
        memory_context = torch.einsum("bk,bkd->bd", weights, response.memory_state)
        delta = self.context_proj(memory_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta

        scratchpad = response.scratchpad_tokens
        per_hop = response.per_hop_attention
        if scratchpad is None or per_hop is None:
            raise ValueError("MANN response must include scratchpad_tokens and per_hop_attention")
        disagreement = per_hop.var(dim=1).mean(dim=-1)
        visibility = WMMANNTraceVisibility(
            pre_fusion_output=output,
            per_hop_attention=per_hop,
            scratchpad_tokens=scratchpad,
            confidence=response.confidence,
            disagreement=disagreement,
        )
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_mann_cross_attention",
            "memory_type": "mann",
            "mann_trace_visibility": visibility.to_dict(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_mann_cross_attention",
                "confidence": float(response.confidence.mean().detach().cpu()) if finite else 0.0,
                "disagreement": float(disagreement.mean().detach().cpu()),
                "mann_trace_visible": True,
                "scratchpad_visible": True,
                "per_hop_attention_visible": True,
            },
        }
        out = WMMANNCrossAttentionOutput(output=output, memory_context=memory_context, response=response, visibility=visibility, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output


# ---------------------------------------------------------------------------
# WM-QD-4A external-memory/shared-slot/QH quality contract
# ---------------------------------------------------------------------------

def wm_qd4a_external_memory_contract() -> dict:
    """Return serialization-safe quality metadata for this external-memory layer.

    This no-mutation contract declares external memory response schemas, MANN
    trace visibility, fusion shape checks, shared-slot ownership/conflict
    metadata, QH code schema validation, interference checks, trace
    serialization, PAAMA-X write-permission metadata, fallback behavior, and
    compatibility with QDTWorkingMemory.
    """
    return external_memory_contract_trace(module=__name__)
