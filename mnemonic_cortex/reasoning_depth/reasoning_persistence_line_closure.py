"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning persistence line closure.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class PersistenceLineClosureError(ValueError):
    """Raised when persistence-line closure metadata is unsafe."""


class PersistenceLineDecision(str, Enum):
    CLOSED_METADATA_ONLY = "closed_metadata_only"
    CONTINUE_WITH_EXPLICIT_BACKEND_STAGE = "continue_with_explicit_backend_stage"
    BLOCKED_UNSAFE = "blocked_unsafe"


@dataclass(frozen=True)
class PersistenceLineClosureConfig:
    """Final closure config for the persistence design line.

    This stage closes the metadata-only persistence line unless a future command
    explicitly authorizes real backend implementation. It does not enable writes.
    """

    enabled: bool = False
    allow_real_backend_next: bool = False
    require_explicit_future_authorization: bool = True
    require_no_real_writes: bool = True
    require_json_safe_report: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise PersistenceLineClosureError("no_mutation_by_default must remain true")
        if not self.require_no_real_writes:
            raise PersistenceLineClosureError("require_no_real_writes must remain true")
        if not self.require_explicit_future_authorization:
            raise PersistenceLineClosureError("future backend authorization must be explicit")

    @classmethod
    def disabled(cls) -> "PersistenceLineClosureConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PersistenceLineClosureConfig":
        return cls(enabled=True)


@dataclass
class PersistenceDecisionRecord:
    decision_id: str
    decision: PersistenceLineDecision
    reason: str
    allowed_next_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "decision_id": self.decision_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "allowed_next_actions": list(self.allowed_next_actions),
            "blocked_actions": list(self.blocked_actions),
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class PersistenceLineClosureReport:
    enabled: bool
    decision_record: Dict[str, Any]
    closed_stages: List[str]
    remaining_deferred_work: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"persistence_line_closure_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "decision_record": _safe_jsonable(self.decision_record),
            "closed_stages": list(self.closed_stages),
            "remaining_deferred_work": list(self.remaining_deferred_work),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "audit_metadata": True,
                "write_permission_hooks": True,
                "policy_lane_integration": True,
                "quarantine_hooks": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class PersistenceLineClosure:
    """Builds final closure metadata for REASON-4A/4B persistence line."""

    def __init__(self, config: Optional[PersistenceLineClosureConfig] = None):
        self.config = config or PersistenceLineClosureConfig.disabled()
        self.config.validate()

    def close(self, *, lineage: Optional[Dict[str, Any]] = None) -> PersistenceLineClosureReport:
        lineage = lineage or {}
        if not self.config.enabled:
            decision = PersistenceDecisionRecord(
                decision_id=f"persistence_decision_{uuid.uuid4().hex[:16]}",
                decision=PersistenceLineDecision.BLOCKED_UNSAFE,
                reason="closure disabled; no persistence action permitted",
                blocked_actions=self._blocked_actions() + ["closure_disabled"],
                lineage=lineage,
            )
            return self._report(False, decision, lineage)

        if self.config.allow_real_backend_next:
            decision_value = PersistenceLineDecision.CONTINUE_WITH_EXPLICIT_BACKEND_STAGE
            reason = "future real backend stage may proceed only under a separate explicit authorization command"
            allowed_next = ["explicitly_authorized_backend_implementation_command"]
        else:
            decision_value = PersistenceLineDecision.CLOSED_METADATA_ONLY
            reason = "metadata-only persistence design line closed; real writes remain deferred"
            allowed_next = ["final_closure", "separate_future_backend_authorization_only"]

        decision = PersistenceDecisionRecord(
            decision_id=f"persistence_decision_{uuid.uuid4().hex[:16]}",
            decision=decision_value,
            reason=reason,
            allowed_next_actions=allowed_next,
            blocked_actions=self._blocked_actions(),
            lineage=lineage,
        )
        return self._report(True, decision, lineage)

    def _report(self, enabled: bool, decision: PersistenceDecisionRecord, lineage: Dict[str, Any]) -> PersistenceLineClosureReport:
        report = PersistenceLineClosureReport(
            enabled=enabled,
            decision_record=decision.to_dict(),
            closed_stages=["REASON-4A", "REASON-4B", "REASON-4C"],
            remaining_deferred_work=[
                "real persistence backend implementation requires explicit future authorization",
                "production store credentials and dependency checks are not implemented",
                "actual memory-store mutation remains disabled",
            ],
            safety_flags=self._safety_flags(),
            lineage=lineage,
        )
        if self.config.require_json_safe_report:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "automatic_persistence_write",
            "permanent_memory_store_mutation",
            "model_weight_mutation",
            "optimizer_mutation",
            "destructive_wm_mann_ltm_replacement",
            "hidden_backend_activation",
            "fake_production_complete_claim",
        ]

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "automatic_persistence": False,
            "real_store_write_performed": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
            "fake_production_complete_claim": False,
        }


def reasoning_persistence_line_closure_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_persistence_line_closure",
        "stage": "REASON-4C",
        "default_enabled": False,
        "metadata_line_closure": True,
        "real_store_write_performed": False,
        "future_backend_requires_explicit_authorization": True,
        "json_safe_report": True,
    }
