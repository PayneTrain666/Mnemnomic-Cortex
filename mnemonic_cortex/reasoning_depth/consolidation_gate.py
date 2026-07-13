"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: consolidation gate.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time
import uuid

from .reasoning_orchestration_trace import _safe_jsonable


class ConsolidationGateError(ValueError):
    """Raised when a consolidation decision cannot be evaluated safely."""


class ConsolidationDecision(str, Enum):
    SHADOW_ONLY = "shadow_only"
    DENIED = "denied"
    QUARANTINED = "quarantined"
    COMMIT_READY = "commit_ready"


@dataclass(frozen=True)
class ConsolidationGateConfig:
    """Safe default gate config.

    REASON-2A does not permanently commit memory. `allow_commit_ready` only
    marks a proposal as ready for a later explicit commit path; it does not
    mutate LTM.
    """

    allow_commit_ready: bool = False
    min_confidence: float = 0.65
    max_disagreement: float = 0.35
    quarantine_on_conflict: bool = True
    require_write_permission: bool = True
    max_payload_chars: int = 12000

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ConsolidationGateError("min_confidence must be in [0,1]")
        if not (0.0 <= self.max_disagreement <= 1.0):
            raise ConsolidationGateError("max_disagreement must be in [0,1]")
        if self.max_payload_chars <= 0:
            raise ConsolidationGateError("max_payload_chars must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_commit_ready": self.allow_commit_ready,
            "min_confidence": self.min_confidence,
            "max_disagreement": self.max_disagreement,
            "quarantine_on_conflict": self.quarantine_on_conflict,
            "require_write_permission": self.require_write_permission,
            "max_payload_chars": self.max_payload_chars,
        }


@dataclass
class ConsolidationGateEvaluation:
    evaluation_id: str
    decision: ConsolidationDecision
    reason: str
    committed: bool
    shadow_only: bool
    write_permission: bool
    confidence: float
    disagreement: float
    canonical_slot_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "committed": self.committed,
            "shadow_only": self.shadow_only,
            "write_permission": self.write_permission,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "canonical_slot_id": self.canonical_slot_id,
            "payload": _safe_jsonable(self.payload),
            "created_at": self.created_at,
            "paamax_metadata": {
                "write_permission_required": True,
                "write_permission_granted": bool(self.write_permission),
                "conflict_quarantine_hooks": True,
                "audit_metadata": True,
            },
            "safety": {
                "permanent_memory_store_mutation": False,
                "shadow_or_ready_only": True,
            },
        }


@dataclass
class ShadowConsolidationGate:
    """Gates LTM consolidation proposals without performing permanent commits."""

    config: ConsolidationGateConfig = field(default_factory=ConsolidationGateConfig)

    def __post_init__(self) -> None:
        self.config.validate()

    def evaluate(
        self,
        proposal: Optional[Dict[str, Any]],
        *,
        write_permission: bool = False,
        confidence: float = 0.5,
        disagreement: float = 0.0,
        conflict: bool = False,
        canonical_slot_id: Optional[str] = None,
    ) -> ConsolidationGateEvaluation:
        if proposal is None:
            return self._decision(ConsolidationDecision.DENIED, "missing consolidation proposal", write_permission, confidence, disagreement, canonical_slot_id, {})
        if len(repr(proposal)) > self.config.max_payload_chars:
            return self._decision(ConsolidationDecision.DENIED, "proposal payload exceeds bounded gate budget", write_permission, confidence, disagreement, canonical_slot_id, {"payload_truncated": True})
        if conflict and self.config.quarantine_on_conflict:
            return self._decision(ConsolidationDecision.QUARANTINED, "conflict flag raised; proposal quarantined", write_permission, confidence, disagreement, canonical_slot_id, proposal)
        if disagreement > self.config.max_disagreement:
            return self._decision(ConsolidationDecision.DENIED, "disagreement exceeds gate threshold", write_permission, confidence, disagreement, canonical_slot_id, proposal)
        if confidence < self.config.min_confidence:
            return self._decision(ConsolidationDecision.SHADOW_ONLY, "confidence below commit-ready threshold; retained as shadow proposal", write_permission, confidence, disagreement, canonical_slot_id, proposal)
        if self.config.require_write_permission and not write_permission:
            return self._decision(ConsolidationDecision.SHADOW_ONLY, "write permission missing; retained as shadow proposal", write_permission, confidence, disagreement, canonical_slot_id, proposal)
        if self.config.allow_commit_ready:
            return self._decision(ConsolidationDecision.COMMIT_READY, "proposal marked commit-ready for later explicit commit path; no mutation performed", write_permission, confidence, disagreement, canonical_slot_id, proposal)
        return self._decision(ConsolidationDecision.SHADOW_ONLY, "gate default is shadow-only; no permanent commit performed", write_permission, confidence, disagreement, canonical_slot_id, proposal)

    def _decision(self, decision: ConsolidationDecision, reason: str, write_permission: bool, confidence: float, disagreement: float, canonical_slot_id: Optional[str], payload: Dict[str, Any]) -> ConsolidationGateEvaluation:
        return ConsolidationGateEvaluation(
            evaluation_id=f"consolidation_eval_{uuid.uuid4().hex[:16]}",
            decision=decision,
            reason=reason,
            committed=False,
            shadow_only=decision in {ConsolidationDecision.SHADOW_ONLY, ConsolidationDecision.COMMIT_READY},
            write_permission=write_permission,
            confidence=float(confidence),
            disagreement=float(disagreement),
            canonical_slot_id=canonical_slot_id,
            payload=payload,
        )


def consolidation_gate_contract() -> Dict[str, Any]:
    return {
        "module": "consolidation_gate",
        "stage": "REASON-2A",
        "default_decision": "shadow_only",
        "permanent_commit": False,
        "commit_ready_is_not_commit": True,
        "conflict_quarantine_hooks": True,
        "write_permission_required": True,
        "bounded_payload": True,
    }
