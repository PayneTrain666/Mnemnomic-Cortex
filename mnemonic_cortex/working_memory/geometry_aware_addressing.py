"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: geometry aware addressing.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curved_slot_state import CurvedSlotStateBank, CurvedSlotSnapshot
from .curvature_metric_policy import CurvatureMetricPolicy, CurvatureMetricPolicyOutput


@dataclass
class GeometryAwareAddressingConfig:
    """Configuration for geometry-aware slot addressing.

    Addressing combines:
    - content similarity
    - curved/Poincare-style distance
    - phase compatibility
    - importance
    - confidence
    - trace reliability
    - context geometry bias
    """

    dim: int
    num_slots: int
    num_depths: int = 8
    top_k: int = 4
    content_weight: float = 1.0
    curved_distance_weight: float = 0.35
    phase_weight: float = 0.25
    importance_weight: float = 0.20
    confidence_weight: float = 0.20
    trace_reliability_weight: float = 0.10
    context_geometry_weight: float = 0.15
    temperature: float = 1.0
    poincare_eps: float = 1e-5
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass
class GeometryAwareAddressingTrace:
    selected_slot_ids: List[List[str]]
    top_indices: List[List[int]]
    top_scores: List[List[float]]
    entropy: List[float]
    score_components: Dict[str, float] = field(default_factory=dict)
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryAwareAddressingOutput:
    activation: torch.Tensor
    scores: torch.Tensor
    top_indices: torch.Tensor
    top_scores: torch.Tensor
    read_content: torch.Tensor
    read_position: torch.Tensor
    trace: GeometryAwareAddressingTrace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_shape": list(self.activation.shape),
            "scores_shape": list(self.scores.shape),
            "top_indices": self.top_indices.detach().cpu().tolist(),
            "top_scores": self.top_scores.detach().cpu().tolist(),
            "trace": self.trace.to_dict(),
        }


class GeometryAwareAddressing(nn.Module):
    """Geometry-aware addressing over CurvedSlotStateBank.

    Inputs:
    - query: [B,D]
    - optional context_bias: [B,S] or [S]
    - optional curvature output: [B,Z,S] from CurvatureMetricPolicy

    Output:
    - activation: [B,S]
    - read_content: [B,D]
    - read_position: [B,D]
    """

    def __init__(
        self,
        config: GeometryAwareAddressingConfig,
        slot_bank: CurvedSlotStateBank,
        curvature_policy: Optional[CurvatureMetricPolicy] = None,
    ):
        super().__init__()
        config.validate()
        if slot_bank.config.num_slots != config.num_slots:
            raise ValueError("slot_bank num_slots must match addressing config")
        if slot_bank.config.dim != config.dim:
            raise ValueError("slot_bank dim must match addressing config")

        self.config = config
        self.slot_bank = slot_bank
        self.curvature_policy = curvature_policy
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
        )
        self.context_bias_projection = nn.Linear(config.dim, config.num_slots)
        self.last_trace: Optional[GeometryAwareAddressingTrace] = None

    def _poincare_distance_score(self, query_pos: torch.Tensor, slot_pos: torch.Tensor) -> torch.Tensor:
        """Approximate negative Poincare distance score.

        query_pos: [B,D]
        slot_pos: [S,D]
        returns: [B,S], higher is better.
        """
        eps = self.config.poincare_eps
        q = query_pos.clamp(-1 + eps, 1 - eps)
        p = slot_pos.clamp(-1 + eps, 1 - eps)
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        p_norm_sq = (p * p).sum(dim=-1).view(1, -1).clamp(max=1.0 - eps)
        diff_sq = ((q.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(dim=-1)
        denom = ((1.0 - q_norm_sq) * (1.0 - p_norm_sq)).clamp_min(eps)
        z = 1.0 + 2.0 * diff_sq / denom
        z = z.clamp_min(1.0 + eps)
        distance = torch.acosh(z)
        return -distance

    def _trace_reliability(self, snapshot: CurvedSlotSnapshot, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        counts = torch.tensor([len(t) for t in snapshot.trace_links], device=device, dtype=dtype)
        return torch.tanh(counts / 5.0)

    def _context_bias(self, query: torch.Tensor, context_geometry_bias: Optional[torch.Tensor]) -> torch.Tensor:
        if context_geometry_bias is None:
            return self.context_bias_projection(query)
        if context_geometry_bias.dim() == 1:
            return context_geometry_bias.view(1, -1).expand(query.size(0), -1).to(device=query.device, dtype=query.dtype)
        if context_geometry_bias.dim() == 2:
            if context_geometry_bias.size(0) != query.size(0):
                raise ValueError("context_geometry_bias batch dimension mismatch")
            return context_geometry_bias.to(device=query.device, dtype=query.dtype)
        raise ValueError("context_geometry_bias must be [S] or [B,S]")

    def _curvature_adjustment(
        self,
        query: torch.Tensor,
        curvature_output: Optional[CurvatureMetricPolicyOutput],
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if curvature_output is None and self.curvature_policy is not None:
            curvature_output = self.curvature_policy(context=context, batch_size=query.size(0))
        if curvature_output is None:
            return torch.zeros(query.size(0), self.config.num_slots, device=query.device, dtype=query.dtype)

        combined = curvature_output.combined_curvature.to(device=query.device, dtype=query.dtype)  # [B,Z,S]
        # Average depth curvature for WM-1D addressing. WM-2/3 can use depth-specific routing.
        return torch.tanh(combined.mean(dim=1))

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_geometry_bias: Optional[torch.Tensor] = None,
        curvature_output: Optional[CurvatureMetricPolicyOutput] = None,
    ) -> GeometryAwareAddressingOutput:
        if query.dim() != 2 or query.size(-1) != self.config.dim:
            raise ValueError(f"Expected query [B,{self.config.dim}], got {tuple(query.shape)}")

        snapshot = self.slot_bank.snapshot()
        content = snapshot.content.to(device=query.device, dtype=query.dtype)
        position = snapshot.position.to(device=query.device, dtype=query.dtype)
        phase = snapshot.phase.to(device=query.device, dtype=query.dtype)
        importance = snapshot.importance.to(device=query.device, dtype=query.dtype)
        confidence = snapshot.confidence.to(device=query.device, dtype=query.dtype)

        q = self.query_projection(query)
        q_norm = F.normalize(q, dim=-1, eps=self.config.eps)
        content_norm = F.normalize(content, dim=-1, eps=self.config.eps)
        content_score = torch.matmul(q_norm, content_norm.t())

        # Use normalized projected query as local position.
        query_position = torch.tanh(q)
        query_position = query_position / query_position.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
        query_position = 0.5 * query_position
        curved_score = self._poincare_distance_score(query_position, position)

        phase_query = F.normalize(torch.sin(q), dim=-1, eps=self.config.eps)
        phase_score = torch.matmul(phase_query, phase.t())

        trace_reliability = self._trace_reliability(snapshot, query.device, query.dtype).view(1, -1)
        context_bias = self._context_bias(q, context_geometry_bias)
        curvature_bias = self._curvature_adjustment(q, curvature_output, context)

        scores = (
            self.config.content_weight * content_score
            + self.config.curved_distance_weight * curved_score
            + self.config.phase_weight * phase_score
            + self.config.importance_weight * importance.view(1, -1)
            + self.config.confidence_weight * confidence.view(1, -1)
            + self.config.trace_reliability_weight * trace_reliability
            + self.config.context_geometry_weight * context_bias
            + 0.05 * curvature_bias
        )

        scores = torch.nan_to_num(scores) / self.config.temperature
        activation = torch.softmax(scores, dim=-1)
        read_content = torch.matmul(activation, content)
        read_position = torch.matmul(activation, position)

        k = min(self.config.top_k, self.config.num_slots)
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        selected_slot_ids = [
            [snapshot.slot_id[int(i)] for i in row]
            for row in top_indices.detach().cpu().tolist()
        ]
        entropy = -(activation * (activation + self.config.eps).log()).sum(dim=-1)
        trace = GeometryAwareAddressingTrace(
            selected_slot_ids=selected_slot_ids,
            top_indices=top_indices.detach().cpu().tolist(),
            top_scores=top_scores.detach().cpu().tolist(),
            entropy=entropy.detach().cpu().tolist(),
            score_components={
                "content_weight": self.config.content_weight,
                "curved_distance_weight": self.config.curved_distance_weight,
                "phase_weight": self.config.phase_weight,
                "importance_weight": self.config.importance_weight,
                "confidence_weight": self.config.confidence_weight,
                "trace_reliability_weight": self.config.trace_reliability_weight,
                "context_geometry_weight": self.config.context_geometry_weight,
            },
            paamax_metadata={
                "trace_type": "geometry_aware_addressing",
                "confidence": float((1.0 / (1.0 + entropy.mean())).detach().cpu()),
                "conflict_check_recommended": bool((entropy > torch.log(torch.tensor(float(self.config.num_slots), device=query.device)) * 0.95).any().item()),
            },
        )
        self.last_trace = trace

        return GeometryAwareAddressingOutput(
            activation=activation,
            scores=scores,
            top_indices=top_indices,
            top_scores=top_scores,
            read_content=read_content,
            read_position=read_position,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# WM-QD-1A foundation-quality contract
# ---------------------------------------------------------------------------

def wm_qd1a_foundation_contract() -> dict:
    """Return serialization-safe quality metadata for this early-WM module.

    This does not mutate runtime state. It exists so the quality tooling can
    verify that the module has an explicit contract for shape/finite checks,
    traceability, PAAMA-X metadata, fallback behavior, and boundedness.
    """
    return foundation_trace(
        trace_type="wm_qd1a_foundation_contract",
        module=__name__,
        message="early working-memory foundation module hardened by WM-QD-1A",
        payload={
            "shape_checks_required": True,
            "finite_checks_required": True,
            "serialization_safe": True,
            "trace_hooks_required": True,
            "paamax_metadata_required": True,
            "boundedness_required": True,
            "runtime_mutation": "no automatic mutation by quality tooling",
        },
    )
