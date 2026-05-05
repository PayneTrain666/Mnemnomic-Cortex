from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curved_slot_state import CurvedSlotStateBank


LANE_NAMES = (
    "vector",
    "hyperbolic",
    "temporal",
    "spatial",
    "procedural",
    "trace",
    "policy",
)


@dataclass
class RetrievalLaneConfig:
    """Configuration for memory retrieval lanes.

    Query contract:
    - query: [B,D]

    Lane output contract:
    - candidates: [B,K,D]
    - scores: [B,K]
    """

    dim: int
    top_k: int = 4
    enabled_lanes: Tuple[str, ...] = LANE_NAMES
    hyperbolic_radius: float = 0.995
    temporal_decay: float = 0.05
    policy_floor_score: float = 0.10
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not (0.0 < self.hyperbolic_radius < 1.0):
            raise ValueError("hyperbolic_radius must be in (0,1)")
        unknown = [lane for lane in self.enabled_lanes if lane not in LANE_NAMES]
        if unknown:
            raise ValueError(f"unknown lanes: {unknown}")


@dataclass
class RetrievalLaneOutput:
    lane_name: str
    candidates: torch.Tensor
    scores: torch.Tensor
    slot_indices: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_name": self.lane_name,
            "candidates_shape": list(self.candidates.shape),
            "scores_shape": list(self.scores.shape),
            "slot_indices": self.slot_indices.detach().cpu().tolist(),
            "metadata": self.metadata,
        }


@dataclass
class WMRetrievalLanesOutput:
    query_shape: list
    lane_outputs: Dict[str, RetrievalLaneOutput]
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_shape": self.query_shape,
            "lane_outputs": {name: out.to_dict() for name, out in self.lane_outputs.items()},
            "paamax_metadata": self.paamax_metadata,
        }


class WMRetrievalLanes(nn.Module):
    """Run multiple small-k retrieval lanes for QDT-WM.

    Lanes:
    - vector: cosine similarity against slot content.
    - hyperbolic: Poincare-style negative distance against slot positions.
    - temporal: recency/last_updated weighted retrieval.
    - spatial: position/tangent proximity proxy.
    - procedural: phase/direction compatibility proxy.
    - trace: trace-link count/reliability retrieval.
    - policy: governance-safe retrieval with policy floor metadata.
    """

    def __init__(
        self,
        config: RetrievalLaneConfig,
        slot_bank: CurvedSlotStateBank,
    ):
        super().__init__()
        config.validate()
        if slot_bank.config.dim != config.dim:
            raise ValueError("slot_bank dim must match RetrievalLaneConfig dim")
        self.config = config
        self.slot_bank = slot_bank
        self.query_norm = nn.LayerNorm(config.dim)
        self.lane_query = nn.ModuleDict({lane: nn.Linear(config.dim, config.dim) for lane in LANE_NAMES})
        self.last_output: Optional[WMRetrievalLanesOutput] = None

    def _validate_query(self, query: torch.Tensor) -> None:
        if query.dim() != 2 or query.size(-1) != self.config.dim:
            raise ValueError(f"Expected query [B,{self.config.dim}], got {tuple(query.shape)}")
        if not torch.isfinite(query).all():
            raise ValueError("query contains NaN or Inf")

    def _topk(self, scores: torch.Tensor, candidates_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = min(self.config.top_k, candidates_base.size(0))
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        expanded = candidates_base.index_select(0, top_idx.reshape(-1)).reshape(scores.size(0), k, -1)
        return expanded, top_scores, top_idx

    def _poincare_score(self, query: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        eps = self.config.eps
        q = torch.tanh(query)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        q = self.config.hyperbolic_radius * 0.5 * q
        p = positions.clamp(-self.config.hyperbolic_radius, self.config.hyperbolic_radius)
        q_norm = (q * q).sum(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        p_norm = (p * p).sum(dim=-1).view(1, -1).clamp(max=1.0 - eps)
        diff_sq = ((q.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(dim=-1)
        denom = ((1.0 - q_norm) * (1.0 - p_norm)).clamp_min(eps)
        z = (1.0 + 2.0 * diff_sq / denom).clamp_min(1.0 + eps)
        return -torch.acosh(z)

    def _lane_scores(self, lane: str, query: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        snap = self.slot_bank.snapshot()
        content = snap.content.to(device=query.device, dtype=query.dtype)
        position = snap.position.to(device=query.device, dtype=query.dtype)
        tangent = snap.tangent.to(device=query.device, dtype=query.dtype)
        phase = snap.phase.to(device=query.device, dtype=query.dtype)
        importance = snap.importance.to(device=query.device, dtype=query.dtype)
        confidence = snap.confidence.to(device=query.device, dtype=query.dtype)
        last_updated = snap.last_updated.to(device=query.device, dtype=query.dtype)

        q = self.lane_query[lane](self.query_norm(query))
        q_norm = F.normalize(q, dim=-1, eps=self.config.eps)

        if lane == "vector":
            return torch.matmul(q_norm, F.normalize(content, dim=-1, eps=self.config.eps).t())
        if lane == "hyperbolic":
            return self._poincare_score(q, position)
        if lane == "temporal":
            recency = last_updated - last_updated.min()
            recency = recency / recency.max().clamp_min(self.config.eps)
            semantic = torch.matmul(q_norm, F.normalize(content, dim=-1, eps=self.config.eps).t())
            return semantic + self.config.temporal_decay * recency.view(1, -1)
        if lane == "spatial":
            spatial_base = F.normalize(position + 0.25 * tangent, dim=-1, eps=self.config.eps)
            return torch.matmul(q_norm, spatial_base.t())
        if lane == "procedural":
            proc_base = F.normalize(phase + 0.25 * tangent, dim=-1, eps=self.config.eps)
            return torch.matmul(torch.sin(q_norm), proc_base.t())
        if lane == "trace":
            trace_counts = torch.tensor([len(t) for t in snap.trace_links], device=query.device, dtype=query.dtype)
            trace_score = torch.tanh(trace_counts / 5.0)
            semantic = torch.matmul(q_norm, F.normalize(content, dim=-1, eps=self.config.eps).t())
            return 0.4 * semantic + trace_score.view(1, -1)
        if lane == "policy":
            # Keep policy lane conservative: content confidence + importance floor.
            semantic = torch.matmul(q_norm, F.normalize(content, dim=-1, eps=self.config.eps).t())
            return 0.2 * semantic + confidence.view(1, -1) + self.config.policy_floor_score * importance.view(1, -1)

        raise ValueError(f"Unsupported lane: {lane}")

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        self._validate_query(query)
        snap = self.slot_bank.snapshot()
        content = snap.content.to(device=query.device, dtype=query.dtype)

        outputs: Dict[str, RetrievalLaneOutput] = {}
        for lane in self.config.enabled_lanes:
            scores = torch.nan_to_num(self._lane_scores(lane, query, context=context))
            candidates, top_scores, top_idx = self._topk(scores, content)
            outputs[lane] = RetrievalLaneOutput(
                lane_name=lane,
                candidates=candidates,
                scores=top_scores,
                slot_indices=top_idx,
                metadata={
                    "trace_type": f"retrieval_lane_{lane}",
                    "top_k": int(top_idx.size(-1)),
                    "finite": bool(torch.isfinite(candidates).all().item() and torch.isfinite(top_scores).all().item()),
                    "paamax_policy_lane": lane == "policy",
                    "write_permission_required": lane == "policy",
                },
            )

        result = WMRetrievalLanesOutput(
            query_shape=list(query.shape),
            lane_outputs=outputs,
            paamax_metadata={
                "trace_type": "wm_retrieval_lanes",
                "enabled_lanes": list(self.config.enabled_lanes),
                "policy_lane_present": "policy" in outputs,
                "write_permission_hooks": "policy" in outputs,
            },
        )
        self.last_output = result
        if return_trace:
            return result, result.to_dict()
        return result

    def stability_report(self, query: torch.Tensor) -> Dict[str, Any]:
        out = self.forward(query)
        finite = all(v.metadata.get("finite", False) for v in out.lane_outputs.values())
        return {
            "ok": bool(finite),
            "finite": bool(finite),
            "lane_count": len(out.lane_outputs),
            "lanes": list(out.lane_outputs.keys()),
        }
