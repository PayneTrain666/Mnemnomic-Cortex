"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm ltm cross attention.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from geometry.chart_native import majority_chart, project_with_residual

from .wm_external_memory_interfaces import ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank
from .wm_native_chart_geometry import charts_for_context_map


@dataclass
class WMLTMCrossAttentionConfig:
    dim: int
    top_k: int = 4
    residual_mix: float = 0.15
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMLTMCrossAttentionOutput:
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


class WMLTMCrossAttention(nn.Module):
    """Long-term-memory cross attention.

    LTM role:
    - stable canonical/factual memory.
    - weighted more for high-confidence context.
    """

    def __init__(self, config: WMLTMCrossAttentionConfig, external_bank: Optional[SyntheticExternalMemoryBank] = None):
        super().__init__()
        config.validate()
        self.config = config
        self.external_bank = external_bank or SyntheticExternalMemoryBank("ltm", config.dim, slots=max(config.top_k, 8))
        self.query_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.context_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMLTMCrossAttentionOutput] = None

    def forward(
        self,
        tokens: torch.Tensor,
        depth_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        context_map_name: Optional[str] = None,
        native_chart_mix: Optional[float] = None,
    ):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        query_state = self.query_proj(tokens.mean(dim=1))
        chart_mix = 0.0 if native_chart_mix is None else float(max(0.0, min(1.0, native_chart_mix)))
        map_name = context_map_name or "hierarchical"
        charts = charts_for_context_map(map_name, 8)
        if chart_mix > 0.0:
            query_state, _ = project_with_residual(query_state, majority_chart(charts), mix=chart_mix)
        request = ExternalMemoryQuery(
            "ltm",
            query_state=query_state,
            depth_state=depth_state,
            context=context if context is not None else tokens,
            metadata={
                "source": "WMLTMCrossAttention",
                "geometry_map": map_name,
                "geometry_by_depth": charts,
                "native_chart_mix": chart_mix,
            },
        )
        response = self.external_bank.query(request, top_k=self.config.top_k)
        weights = torch.softmax(response.scores, dim=-1)
        memory_context = torch.einsum("bk,bkd->bd", weights, response.memory_state)
        delta = self.context_proj(memory_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta
        finite = bool(torch.isfinite(output).all().item())
        trace = {
            "trace_type": "wm_ltm_cross_attention",
            "memory_type": "ltm",
            "pre_fusion_output_shape": list(output.shape),
            "confidence": response.confidence.detach().cpu().tolist(),
            "finite": finite,
            "geometry_map": map_name,
            "native_chart_mix": chart_mix,
            "paamax_metadata": {
                "trace_type": "wm_ltm_cross_attention",
                "confidence": float(response.confidence.mean().detach().cpu()) if finite else 0.0,
                "audit_metadata": {"external_adapter_kind": response.metadata.get("adapter_kind")},
            },
        }
        out = WMLTMCrossAttentionOutput(output=output, memory_context=memory_context, response=response, trace=trace)
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
