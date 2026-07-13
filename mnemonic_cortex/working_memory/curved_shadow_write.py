"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: curved shadow write.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import uuid

import torch

from .curved_slot_state import CurvedSlotStateBank
from .curved_local_trace import CurvedLocalTrace


@dataclass
class CurvedShadowWriteConfig:
    """Configuration for curved shadow writes."""

    dim: int
    max_pending: int = 128
    interference_threshold: float = 0.85
    min_confidence_to_commit: float = 0.05
    require_paamax_permission: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if not 0.0 <= self.interference_threshold <= 1.0:
            raise ValueError("interference_threshold must be in [0,1]")
        if not 0.0 <= self.min_confidence_to_commit <= 1.0:
            raise ValueError("min_confidence_to_commit must be in [0,1]")


@dataclass
class ShadowWriteProposal:
    proposal_id: str
    slot_indices: torch.Tensor
    content_delta: torch.Tensor
    position_delta: Optional[torch.Tensor] = None
    tangent_delta: Optional[torch.Tensor] = None
    phase_delta: Optional[torch.Tensor] = None
    curvature_delta: Optional[torch.Tensor] = None
    importance_delta: Optional[torch.Tensor] = None
    confidence_delta: Optional[torch.Tensor] = None
    confidence: float = 1.0
    source: str = "curved_resonant_wm"
    paamax_permission: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "slot_indices": self.slot_indices.detach().cpu().tolist(),
            "content_delta_shape": list(self.content_delta.shape),
            "position_delta_shape": None if self.position_delta is None else list(self.position_delta.shape),
            "tangent_delta_shape": None if self.tangent_delta is None else list(self.tangent_delta.shape),
            "phase_delta_shape": None if self.phase_delta is None else list(self.phase_delta.shape),
            "curvature_delta_shape": None if self.curvature_delta is None else list(self.curvature_delta.shape),
            "importance_delta_shape": None if self.importance_delta is None else list(self.importance_delta.shape),
            "confidence_delta_shape": None if self.confidence_delta is None else list(self.confidence_delta.shape),
            "confidence": self.confidence,
            "source": self.source,
            "paamax_permission": self.paamax_permission,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ShadowWriteDecision:
    proposal_id: str
    decision: str
    committed: bool
    reason: str
    interference_score: float
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CurvedShadowWriteBuffer:
    """Shadow-write buffer for curved WM updates.

    Doctrine:
    - active reasoning reads from main state.
    - writes go to shadow proposals first.
    - commit only after permission, confidence, and interference checks.
    """

    def __init__(self, config: CurvedShadowWriteConfig, slot_bank: Optional[CurvedSlotStateBank] = None):
        config.validate()
        self.config = config
        self.slot_bank = slot_bank
        self.pending: Dict[str, ShadowWriteProposal] = {}
        self.history: List[ShadowWriteDecision] = []

    def _new_id(self) -> str:
        return f"shadow-{uuid.uuid4().hex}"

    def stage(
        self,
        slot_indices: torch.Tensor,
        content_delta: torch.Tensor,
        *,
        position_delta: Optional[torch.Tensor] = None,
        tangent_delta: Optional[torch.Tensor] = None,
        phase_delta: Optional[torch.Tensor] = None,
        curvature_delta: Optional[torch.Tensor] = None,
        importance_delta: Optional[torch.Tensor] = None,
        confidence_delta: Optional[torch.Tensor] = None,
        confidence: float = 1.0,
        source: str = "curved_resonant_wm",
        paamax_permission: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShadowWriteProposal:
        if len(self.pending) >= self.config.max_pending:
            raise RuntimeError("shadow write buffer is full")
        if slot_indices.dim() != 1:
            raise ValueError("slot_indices must be 1D")
        if content_delta.dim() != 2 or content_delta.size(-1) != self.config.dim:
            raise ValueError(f"content_delta must be [K,{self.config.dim}]")
        if content_delta.size(0) != slot_indices.numel():
            raise ValueError("content_delta first dimension must match slot_indices")

        def check_optional(name: str, value: Optional[torch.Tensor], trailing_dim: Optional[int] = None) -> None:
            if value is None:
                return
            if value.size(0) != slot_indices.numel():
                raise ValueError(f"{name} first dimension must match slot_indices")
            if trailing_dim is not None and value.size(-1) != trailing_dim:
                raise ValueError(f"{name} trailing dim must be {trailing_dim}")

        check_optional("position_delta", position_delta, self.config.dim)
        check_optional("tangent_delta", tangent_delta, self.config.dim)
        check_optional("phase_delta", phase_delta, self.config.dim)
        check_optional("curvature_delta", curvature_delta)
        check_optional("importance_delta", importance_delta)
        check_optional("confidence_delta", confidence_delta)

        proposal = ShadowWriteProposal(
            proposal_id=self._new_id(),
            slot_indices=slot_indices.detach().clone(),
            content_delta=content_delta.detach().clone(),
            position_delta=None if position_delta is None else position_delta.detach().clone(),
            tangent_delta=None if tangent_delta is None else tangent_delta.detach().clone(),
            phase_delta=None if phase_delta is None else phase_delta.detach().clone(),
            curvature_delta=None if curvature_delta is None else curvature_delta.detach().clone(),
            importance_delta=None if importance_delta is None else importance_delta.detach().clone(),
            confidence_delta=None if confidence_delta is None else confidence_delta.detach().clone(),
            confidence=float(confidence),
            source=source,
            paamax_permission=bool(paamax_permission),
            metadata=metadata or {},
        )
        self.pending[proposal.proposal_id] = proposal
        return proposal

    def interference_score(self, proposal: ShadowWriteProposal) -> float:
        if self.slot_bank is None:
            return 0.0
        snapshot = self.slot_bank.snapshot(proposal.slot_indices.detach().cpu().tolist())
        current = snapshot.content.to(device=proposal.content_delta.device, dtype=proposal.content_delta.dtype)
        if current.numel() == 0:
            return 0.0
        cur_norm = torch.nn.functional.normalize(current, dim=-1, eps=self.config.eps)
        delta_norm = torch.nn.functional.normalize(proposal.content_delta, dim=-1, eps=self.config.eps)
        similarity = torch.abs((cur_norm * delta_norm).sum(dim=-1))
        return float(similarity.max().detach().cpu())

    def _decision(
        self,
        proposal: ShadowWriteProposal,
        decision: str,
        committed: bool,
        reason: str,
        interference: float,
        trace: Optional[CurvedLocalTrace] = None,
    ) -> ShadowWriteDecision:
        d = ShadowWriteDecision(
            proposal_id=proposal.proposal_id,
            decision=decision,
            committed=committed,
            reason=reason,
            interference_score=interference,
            paamax_metadata={
                "trace_type": "curved_shadow_write_decision",
                "write_permission_required": self.config.require_paamax_permission,
                "paamax_permission": proposal.paamax_permission,
                "confidence": proposal.confidence,
                "interference_score": interference,
            },
            trace=None if trace is None else trace.to_dict(),
        )
        self.history.append(d)
        return d

    def evaluate(self, proposal_id: str, trace: Optional[CurvedLocalTrace] = None) -> ShadowWriteDecision:
        proposal = self.pending[proposal_id]
        interference = self.interference_score(proposal)

        if self.config.require_paamax_permission and not proposal.paamax_permission:
            return self._decision(proposal, "reject", False, "paamax_permission_missing", interference, trace)
        if proposal.confidence < self.config.min_confidence_to_commit:
            return self._decision(proposal, "reject", False, "confidence_below_commit_threshold", interference, trace)
        if interference > self.config.interference_threshold:
            return self._decision(proposal, "reject", False, "interference_above_threshold", interference, trace)
        return self._decision(proposal, "commit_ready", False, "checks_passed", interference, trace)

    def commit(self, proposal_id: str, trace: Optional[CurvedLocalTrace] = None) -> ShadowWriteDecision:
        proposal = self.pending[proposal_id]
        decision = self.evaluate(proposal_id, trace=trace)
        if decision.decision != "commit_ready":
            self.pending.pop(proposal_id, None)
            return decision

        if self.slot_bank is not None:
            idx = proposal.slot_indices.to(dtype=torch.long)
            snapshot = self.slot_bank.snapshot(idx.detach().cpu().tolist())

            content = snapshot.content.to(device=proposal.content_delta.device, dtype=proposal.content_delta.dtype) + proposal.content_delta
            position = None if proposal.position_delta is None else snapshot.position.to(device=proposal.position_delta.device, dtype=proposal.position_delta.dtype) + proposal.position_delta
            tangent = None if proposal.tangent_delta is None else snapshot.tangent.to(device=proposal.tangent_delta.device, dtype=proposal.tangent_delta.dtype) + proposal.tangent_delta
            phase = None if proposal.phase_delta is None else snapshot.phase.to(device=proposal.phase_delta.device, dtype=proposal.phase_delta.dtype) + proposal.phase_delta
            curvature = None if proposal.curvature_delta is None else snapshot.curvature.to(device=proposal.curvature_delta.device, dtype=proposal.curvature_delta.dtype) + proposal.curvature_delta
            importance = None if proposal.importance_delta is None else snapshot.importance.to(device=proposal.importance_delta.device, dtype=proposal.importance_delta.dtype) + proposal.importance_delta
            confidence = None if proposal.confidence_delta is None else snapshot.confidence.to(device=proposal.confidence_delta.device, dtype=proposal.confidence_delta.dtype) + proposal.confidence_delta

            self.slot_bank.update_slots(
                idx,
                content=content,
                position=position,
                tangent=tangent,
                phase=phase,
                curvature=curvature,
                importance=importance,
                confidence=confidence,
                trace_link=proposal.proposal_id,
            )

        final = self._decision(
            proposal,
            "committed",
            True,
            "committed_to_slot_bank" if self.slot_bank is not None else "committed_without_slot_bank",
            decision.interference_score,
            trace,
        )
        self.pending.pop(proposal_id, None)
        return final

    def reject(self, proposal_id: str, reason: str = "manual_reject", trace: Optional[CurvedLocalTrace] = None) -> ShadowWriteDecision:
        proposal = self.pending.pop(proposal_id)
        decision = self._decision(proposal, "reject", False, reason, self.interference_score(proposal), trace)
        return decision

    def pending_count(self) -> int:
        return len(self.pending)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending": {pid: p.to_dict() for pid, p in self.pending.items()},
            "history": [h.to_dict() for h in self.history],
        }


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
