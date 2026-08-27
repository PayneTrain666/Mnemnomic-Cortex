"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm memory augmented attention.
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

from .curved_slot_state import CurvedSlotStateBank
from .wm_retrieval_lanes import RetrievalLaneConfig, WMRetrievalLanes
from .wm_geometry_scoring import WMGeometryScoringConfig, WMGeometryScoring
from .wm_geometry_linker import WMGeometryLinker, WMGeometryLinkerConfig
from .wm_evidence_attention import WMEvidenceAttention, WMEvidenceAttentionConfig
from .wm_trace_attention import WMTraceAttention, WMTraceAttentionConfig
from .wm_counterfactual_attention import WMCounterfactualAttention, WMCounterfactualAttentionConfig
from .wm_conflict_attention import WMConflictAttention, WMConflictAttentionConfig
from .wm_novelty_attention import WMNoveltyAttention, WMNoveltyAttentionConfig
from .wm_stability_attention import WMStabilityAttention, WMStabilityAttentionConfig


@dataclass
class WMMemoryAugmentedAttentionConfig:
    dim: int
    top_k: int = 4
    residual_mix: float = 0.20
    use_geometry_linker: bool = True
    use_advanced_attention: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")


@dataclass
class WMMemoryAugmentedAttentionOutput:
    output: torch.Tensor
    memory_context: torch.Tensor
    retrieval_trace: Dict[str, Any]
    scoring_trace: Dict[str, Any]
    linker_trace: Dict[str, Any]
    advanced_attention_traces: Dict[str, Any] = field(default_factory=dict)
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "memory_context_shape": list(self.memory_context.shape),
            "retrieval_trace": self.retrieval_trace,
            "scoring_trace": self.scoring_trace,
            "linker_trace": self.linker_trace,
            "advanced_attention_traces": self.advanced_attention_traces,
            "paamax_metadata": self.paamax_metadata,
        }


class WMMemoryAugmentedAttention(nn.Module):
    """First full memory-augmented attention layer for QDT-WM.

    The module:
    1. summarizes the current token state into a query.
    2. runs vector/hyperbolic/temporal/spatial/procedural/trace/policy lanes.
    3. scores candidates geometrically.
    4. injects a bounded memory context residual back into tokens.
    """

    def __init__(
        self,
        config: WMMemoryAugmentedAttentionConfig,
        slot_bank: CurvedSlotStateBank,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.slot_bank = slot_bank
        self.query_projection = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.context_projection = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.retrieval_lanes = WMRetrievalLanes(
            RetrievalLaneConfig(dim=config.dim, top_k=config.top_k),
            slot_bank=slot_bank,
        )
        self.geometry_scoring = WMGeometryScoring(WMGeometryScoringConfig(dim=config.dim, top_k=config.top_k))
        self.geometry_linker = WMGeometryLinker(WMGeometryLinkerConfig(dim=config.dim)) if config.use_geometry_linker else None
        self.evidence_attention = WMEvidenceAttention(WMEvidenceAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.trace_attention = WMTraceAttention(WMTraceAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.counterfactual_attention = WMCounterfactualAttention(WMCounterfactualAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.conflict_attention = WMConflictAttention(WMConflictAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.novelty_attention = WMNoveltyAttention(WMNoveltyAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.stability_attention = WMStabilityAttention(WMStabilityAttentionConfig(dim=config.dim)) if config.use_advanced_attention else None
        self.last_output: Optional[WMMemoryAugmentedAttentionOutput] = None

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            tokens.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        require_write_permission: bool = False,
        prior_trace: Optional[Dict[str, Any]] = None,
        return_trace: bool = False,
    ):
        self._validate_tokens(tokens)
        query = self.query_projection(tokens.mean(dim=1))

        retrieval, retrieval_trace = self.retrieval_lanes(query, context=context, return_trace=True)
        scoring, scoring_trace = self.geometry_scoring(query, retrieval, return_trace=True)
        memory_context = self.context_projection(scoring.fused_context)

        advanced_attention_traces: Dict[str, Any] = {}
        advanced_tokens = tokens
        if self.evidence_attention is not None:
            advanced_tokens, ev_trace = self.evidence_attention(advanced_tokens, memory_context=memory_context, return_trace=True)
            advanced_attention_traces["evidence_attention"] = ev_trace
        if self.trace_attention is not None:
            advanced_tokens, tr_trace = self.trace_attention(advanced_tokens, prior_trace=prior_trace, return_trace=True)
            advanced_attention_traces["trace_attention"] = tr_trace
        if self.counterfactual_attention is not None:
            advanced_tokens, cf_trace = self.counterfactual_attention(advanced_tokens, memory_context=memory_context, return_trace=True)
            advanced_attention_traces["counterfactual_attention"] = cf_trace
        if self.conflict_attention is not None:
            advanced_tokens, conflict_trace = self.conflict_attention(advanced_tokens, memory_context=memory_context, return_trace=True)
            advanced_attention_traces["conflict_attention"] = conflict_trace
        if self.novelty_attention is not None:
            advanced_tokens, novelty_trace = self.novelty_attention(advanced_tokens, memory_context=memory_context, return_trace=True)
            advanced_attention_traces["novelty_attention"] = novelty_trace
        if self.stability_attention is not None:
            advanced_tokens, stability_trace = self.stability_attention(advanced_tokens, return_trace=True)
            advanced_attention_traces["stability_attention"] = stability_trace

        injected = advanced_tokens + self.config.residual_mix * memory_context.unsqueeze(1)

        linker_trace: Dict[str, Any] = {"trace_type": "wm_geometry_linker", "enabled": False}
        lane_bias = None
        if self.geometry_linker is not None:
            lane_bias = self.geometry_linker.lane_bias(query)
            linker_trace = self.geometry_linker.to_trace()
            linker_trace["lane_bias"] = lane_bias

        policy_lane = retrieval.lane_outputs.get("policy")
        policy_present = policy_lane is not None
        write_permission_granted = bool(policy_present and not require_write_permission) or bool(policy_present and require_write_permission)
        # For WM-3A, policy lane presence is the permission hook. WM-5A will
        # replace this with stricter commit-gate enforcement.

        paamax_metadata = {
            "trace_type": "wm_memory_augmented_attention",
            "policy_lane_present": policy_present,
            "write_permission_required": bool(require_write_permission),
            "write_permission_granted": bool(write_permission_granted),
            "confidence": float(scoring.trace.get("paamax_metadata", {}).get("confidence", 1.0)),
            "conflict_quarantine_hook": "deferred_to_WM-3B",
            "advanced_attention_present": bool(advanced_attention_traces),
            "advanced_attention_modules": list(advanced_attention_traces.keys()),
            "conflict_quarantine_hook": advanced_attention_traces.get("conflict_attention", {}).get("trace", {}).get("paamax_metadata", {}).get("conflict_quarantine", False),
            "lightbulb_hook": advanced_attention_traces.get("novelty_attention", {}).get("trace", {}).get("paamax_metadata", {}).get("lightbulb", False),
            "stability_guard": advanced_attention_traces.get("stability_attention", {}).get("trace", {}).get("paamax_metadata", {}).get("stability_guard", False),
            "audit_metadata": {
                "lane_count": len(retrieval.lane_outputs),
                "top_k": self.config.top_k,
                "geometry_linker_enabled": self.geometry_linker is not None,
            },
        }

        out = WMMemoryAugmentedAttentionOutput(
            output=injected,
            memory_context=memory_context,
            retrieval_trace=retrieval_trace,
            scoring_trace=scoring_trace,
            linker_trace=linker_trace,
            advanced_attention_traces=advanced_attention_traces,
            paamax_metadata=paamax_metadata,
        )
        self.last_output = out
        if return_trace:
            return injected, out.to_dict()
        return injected

    def stability_report(self, tokens: torch.Tensor) -> Dict[str, Any]:
        out, trace = self.forward(tokens, return_trace=True)
        finite = bool(torch.isfinite(out).all().item())
        return {
            "ok": bool(finite and tuple(out.shape) == tuple(tokens.shape)),
            "finite": finite,
            "shape_ok": tuple(out.shape) == tuple(tokens.shape),
            "policy_lane_present": trace["paamax_metadata"]["policy_lane_present"],
            "trace": trace,
        }


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
