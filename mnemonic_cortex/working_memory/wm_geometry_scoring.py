"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm geometry scoring.
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

from .wm_retrieval_lanes import WMRetrievalLanesOutput, LANE_NAMES


@dataclass
class WMGeometryScoringConfig:
    dim: int
    top_k: int = 4
    score_temperature: float = 1.0
    include_lane_priors: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be positive")


@dataclass
class WMGeometryScoringOutput:
    candidates: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor
    fused_context: torch.Tensor
    lane_names: list
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_shape": list(self.candidates.shape),
            "scores_shape": list(self.scores.shape),
            "weights_shape": list(self.weights.shape),
            "fused_context_shape": list(self.fused_context.shape),
            "lane_names": self.lane_names,
            "trace": self.trace,
        }


class WMGeometryScoring(nn.Module):
    """Score and fuse retrieved candidates across geometry lanes."""

    def __init__(self, config: WMGeometryScoringConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.query_projection = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.candidate_projection = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        # Priors are aligned to LANE_NAMES and softly added when lane exists.
        self.lane_prior_logits = nn.Parameter(torch.zeros(len(LANE_NAMES)))
        self.last_output: Optional[WMGeometryScoringOutput] = None

    def _validate_query(self, query: torch.Tensor) -> None:
        if query.dim() != 2 or query.size(-1) != self.config.dim:
            raise ValueError(f"Expected query [B,{self.config.dim}], got {tuple(query.shape)}")
        if not torch.isfinite(query).all():
            query.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, query: torch.Tensor, retrieval: WMRetrievalLanesOutput, return_trace: bool = False):
        self._validate_query(query)
        if not retrieval.lane_outputs:
            raise ValueError("retrieval output has no lane outputs")

        lane_names = list(retrieval.lane_outputs.keys())
        candidates = torch.stack([retrieval.lane_outputs[name].candidates for name in lane_names], dim=1)  # [B,L,K,D]
        lane_scores = torch.stack([retrieval.lane_outputs[name].scores for name in lane_names], dim=1)      # [B,L,K]
        if candidates.dim() != 4 or candidates.size(-1) != self.config.dim:
            raise ValueError("retrieval candidates must be [B,L,K,D]")

        q = F.normalize(self.query_projection(query), dim=-1, eps=self.config.eps)
        c = F.normalize(self.candidate_projection(candidates), dim=-1, eps=self.config.eps)
        semantic = torch.einsum("bd,blkd->blk", q, c)

        scores = semantic + 0.25 * lane_scores
        if self.config.include_lane_priors:
            prior_lookup = {name: i for i, name in enumerate(LANE_NAMES)}
            priors = torch.stack(
                [self.lane_prior_logits[prior_lookup.get(name, 0)] for name in lane_names],
                dim=0,
            ).to(device=query.device, dtype=query.dtype)
            scores = scores + priors.view(1, -1, 1)

        scores = torch.nan_to_num(scores) / self.config.score_temperature
        flat_scores = scores.reshape(scores.size(0), -1)
        weights = torch.softmax(flat_scores, dim=-1).reshape_as(scores)
        fused = (weights.unsqueeze(-1) * candidates).sum(dim=(1, 2))

        finite = bool(torch.isfinite(fused).all().item() and torch.isfinite(weights).all().item())
        trace = {
            "trace_type": "wm_geometry_scoring",
            "candidate_shape": list(candidates.shape),
            "scores_shape": list(scores.shape),
            "lane_names": lane_names,
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_geometry_scoring",
                "confidence": 1.0 if finite else 0.0,
                "policy_lane_included": "policy" in lane_names,
            },
        }
        out = WMGeometryScoringOutput(
            candidates=candidates,
            scores=scores,
            weights=weights,
            fused_context=fused,
            lane_names=lane_names,
            trace=trace,
        )
        self.last_output = out
        if return_trace:
            return out, out.to_dict()
        return out

    def stability_report(self, query: torch.Tensor, retrieval: WMRetrievalLanesOutput) -> Dict[str, Any]:
        out = self.forward(query, retrieval)
        return {
            "ok": bool(out.trace["finite"]),
            "finite": bool(out.trace["finite"]),
            "fused_context_shape": list(out.fused_context.shape),
            "lane_names": out.lane_names,
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
