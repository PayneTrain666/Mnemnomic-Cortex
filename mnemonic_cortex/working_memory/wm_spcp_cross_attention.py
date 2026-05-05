from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wm_external_memory_interfaces import ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank


@dataclass
class WMSPCPCrossAttentionConfig:
    dim: int
    top_k: int = 4
    residual_mix: float = 0.18
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMSPCPCrossAttentionOutput:
    output: torch.Tensor
    memory_context: torch.Tensor
    response: ExternalMemoryResponse
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "memory_context_shape": list(self.memory_context.shape),
            "response": self.response.to_dict(),
            "trace": self.trace,
        }


class WMSPCPCrossAttention(nn.Module):
    """SPCP/procedural cross-attention.

    SPCP role:
    - procedural memory, command chains, workflow/action memory.
    """

    def __init__(self, config: WMSPCPCrossAttentionConfig, external_bank: Optional[SyntheticExternalMemoryBank] = None):
        super().__init__()
        config.validate()
        self.config = config
        self.external_bank = external_bank or SyntheticExternalMemoryBank("spcp", config.dim, slots=max(config.top_k, 8))
        self.query_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.context_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMSPCPCrossAttentionOutput] = None

    def forward(self, tokens: torch.Tensor, depth_state: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, return_trace: bool = False):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contain NaN or Inf")
        query_state = self.query_proj(tokens.mean(dim=1))
        request = ExternalMemoryQuery("spcp", query_state=query_state, depth_state=depth_state, context=context, metadata={"source": "WMSPCPCrossAttention"})
        response = self.external_bank.query(request, top_k=self.config.top_k)
        weights = torch.softmax(response.scores, dim=-1)
        memory_context = torch.einsum("bk,bkd->bd", weights, response.memory_state)
        delta = self.context_proj(memory_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_spcp_cross_attention",
            "memory_type": "spcp",
            "pre_fusion_output_shape": list(output.shape),
            "confidence": response.confidence.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_spcp_cross_attention",
                "confidence": float(response.confidence.mean().detach().cpu()) if finite else 0.0,
                "procedural_memory": True,
            },
        }
        out = WMSPCPCrossAttentionOutput(output=output, memory_context=memory_context, response=response, trace=trace)
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output
