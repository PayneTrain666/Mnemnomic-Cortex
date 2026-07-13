"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning persistence recovery.
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


class PersistenceRecoveryError(ValueError):
    """Raised when persistence recovery metadata is unsafe."""


class RecoveryActionKind(str, Enum):
    NOOP = "noop"
    REPLAY_DRY_RUN = "replay_dry_run"
    ROLLBACK_METADATA = "rollback_metadata"
    QUARANTINE_PAYLOAD = "quarantine_payload"


@dataclass(frozen=True)
class PersistenceRecoveryConfig:
    """Recovery metadata config.

    Recovery is metadata-only in REASON-4B. It prepares plans and rollback
    descriptors without writing or reverting any real store.
    """

    enabled: bool = False
    max_actions: int = 128
    allow_real_rollback: bool = False
    require_quarantine_for_failed_commits: bool = True
    require_json_safe_plan: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_actions <= 0:
            raise PersistenceRecoveryError("max_actions must be positive")
        if self.allow_real_rollback:
            raise PersistenceRecoveryError("real rollback is not allowed in REASON-4B")
        if not self.no_mutation_by_default:
            raise PersistenceRecoveryError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PersistenceRecoveryConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PersistenceRecoveryConfig":
        return cls(enabled=True)


@dataclass
class PersistenceRecoveryAction:
    action_kind: RecoveryActionKind
    reason: str
    related_record_id: str = ""
    related_payload_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: f"persistence_recovery_action_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "action_id": self.action_id,
            "action_kind": self.action_kind.value,
            "reason": self.reason,
            "related_record_id": self.related_record_id,
            "related_payload_id": self.related_payload_id,
            "metadata": _safe_jsonable(self.metadata),
            "real_rollback_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class PersistenceRecoveryPlan:
    enabled: bool
    actions: List[PersistenceRecoveryAction] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: f"persistence_recovery_plan_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "plan_id": self.plan_id,
            "enabled": bool(self.enabled),
            "actions": [action.to_dict() for action in self.actions],
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "real_rollback_performed": False,
            "real_store_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class PersistenceRecoveryPlanner:
    """Builds metadata-only recovery plans for dry-run persistence flows."""

    def __init__(self, config: Optional[PersistenceRecoveryConfig] = None):
        self.config = config or PersistenceRecoveryConfig.disabled()
        self.config.validate()

    def plan(
        self,
        *,
        ledger_payload: Optional[Dict[str, Any]] = None,
        failed_payloads: Optional[List[Dict[str, Any]]] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> PersistenceRecoveryPlan:
        if not self.config.enabled:
            return PersistenceRecoveryPlan(enabled=False, actions=[], safety_flags=self._safety_flags(), lineage=lineage or {})

        actions: List[PersistenceRecoveryAction] = []
        records = []
        if ledger_payload and isinstance(ledger_payload.get("records"), list):
            records = ledger_payload.get("records", [])
        for record in records[: self.config.max_actions]:
            decision = record.get("decision", {})
            if decision.get("approved") is False:
                actions.append(
                    PersistenceRecoveryAction(
                        action_kind=RecoveryActionKind.QUARANTINE_PAYLOAD,
                        reason="denied dry-run decision should remain quarantined",
                        related_record_id=str(record.get("record_id", "")),
                        related_payload_id=str(record.get("request", {}).get("payload", {}).get("payload_id", "")),
                        metadata={"decision_status": decision.get("status")},
                    )
                )

        for payload in (failed_payloads or [])[: max(0, self.config.max_actions - len(actions))]:
            actions.append(
                PersistenceRecoveryAction(
                    action_kind=RecoveryActionKind.QUARANTINE_PAYLOAD if self.config.require_quarantine_for_failed_commits else RecoveryActionKind.NOOP,
                    reason="failed payload recorded for metadata-only recovery review",
                    related_payload_id=str(payload.get("payload_id", "")),
                    metadata={"failure": _safe_jsonable(payload.get("failure", "unspecified"))},
                )
            )

        if not actions:
            actions.append(PersistenceRecoveryAction(RecoveryActionKind.NOOP, "no recovery action required", metadata={"metadata_only": True}))

        plan = PersistenceRecoveryPlan(enabled=True, actions=actions[: self.config.max_actions], safety_flags=self._safety_flags(), lineage=lineage or {})
        if self.config.require_json_safe_plan:
            json.dumps(plan.to_dict(), sort_keys=True)
        return plan

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "metadata_only": True,
            "real_rollback_performed": False,
            "real_store_write_performed": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
        }


def reasoning_persistence_recovery_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_persistence_recovery",
        "stage": "REASON-4B",
        "default_enabled": False,
        "metadata_only": True,
        "real_rollback_performed": False,
        "real_store_write_performed": False,
        "json_safe_plan": True,
    }
