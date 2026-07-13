"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: backend authorization.
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


class BackendAuthorizationError(ValueError):
    """Raised when backend authorization metadata is unsafe or incomplete."""


class BackendAuthorizationStatus(str, Enum):
    NOT_AUTHORIZED = "not_authorized"
    PLAN_ONLY_AUTHORIZED = "plan_only_authorized"
    IMPLEMENTATION_READY_AFTER_REVIEW = "implementation_ready_after_review"
    BLOCKED = "blocked"


class BackendStoreTarget(str, Enum):
    NONE_SELECTED = "none_selected"
    FILESYSTEM_JSONL = "filesystem_jsonl"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    OBJECT_STORE = "object_store"
    VECTOR_STORE = "vector_store"
    GRAPH_STORE = "graph_store"


@dataclass(frozen=True)
class BackendAuthorizationConfig:
    """Authorization-gate config.

    This config authorizes planning only. It does not authorize store writes,
    credential use, schema migration execution, or backend activation.
    """

    enabled: bool = False
    requested_store_target: BackendStoreTarget = BackendStoreTarget.NONE_SELECTED
    explicit_user_authorization: bool = False
    authorize_planning_only: bool = True
    authorize_real_writes: bool = False
    require_second_approval_for_code_that_writes: bool = True
    require_threat_model: bool = True
    require_backup_recovery_plan: bool = True
    require_credential_scope_model: bool = True
    require_migration_plan: bool = True
    require_json_safe_report: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise BackendAuthorizationError("no_mutation_by_default must remain true")
        if self.authorize_real_writes:
            raise BackendAuthorizationError("REASON FUTURE-BACKEND-AUTHORIZATION may not authorize real writes directly")
        if not self.require_second_approval_for_code_that_writes:
            raise BackendAuthorizationError("second approval for write-capable code is required")
        if not self.authorize_planning_only:
            raise BackendAuthorizationError("this stage authorizes planning only")

    @classmethod
    def disabled(cls) -> "BackendAuthorizationConfig":
        return cls(enabled=False)

    @classmethod
    def planning_authorized(cls, target: BackendStoreTarget = BackendStoreTarget.SQLITE) -> "BackendAuthorizationConfig":
        return cls(enabled=True, explicit_user_authorization=True, requested_store_target=target)


@dataclass
class BackendAuthorizationDecision:
    status: BackendAuthorizationStatus
    requested_store_target: BackendStoreTarget
    reason: str
    allowed_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    required_before_write_capable_code: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: f"backend_authorization_decision_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "decision_id": self.decision_id,
            "status": self.status.value,
            "requested_store_target": self.requested_store_target.value,
            "reason": self.reason,
            "allowed_actions": list(self.allowed_actions),
            "blocked_actions": list(self.blocked_actions),
            "required_before_write_capable_code": list(self.required_before_write_capable_code),
            "lineage": _safe_jsonable(self.lineage),
            "real_store_write_authorized": False,
            "write_capable_code_authorized": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class BackendAuthorizationGate:
    """Produces a planning-only backend authorization decision."""

    def __init__(self, config: Optional[BackendAuthorizationConfig] = None):
        self.config = config or BackendAuthorizationConfig.disabled()
        self.config.validate()

    def decide(self, *, lineage: Optional[Dict[str, Any]] = None) -> BackendAuthorizationDecision:
        lineage = lineage or {}
        if not self.config.enabled or not self.config.explicit_user_authorization:
            return BackendAuthorizationDecision(
                status=BackendAuthorizationStatus.NOT_AUTHORIZED,
                requested_store_target=self.config.requested_store_target,
                reason="explicit planning authorization not present",
                allowed_actions=[],
                blocked_actions=self._blocked_actions() + ["planning_not_authorized"],
                required_before_write_capable_code=self._required_before_write_code(),
                lineage=lineage,
            )

        return BackendAuthorizationDecision(
            status=BackendAuthorizationStatus.PLAN_ONLY_AUTHORIZED,
            requested_store_target=self.config.requested_store_target,
            reason="planning and design are authorized; write-capable implementation remains blocked pending second approval",
            allowed_actions=[
                "create_backend_threat_model",
                "create_credential_scope_model",
                "create_backup_recovery_plan",
                "create_migration_plan",
                "create_dry_run_first_implementation_plan",
                "create_tests_for_future_write_gates",
            ],
            blocked_actions=self._blocked_actions(),
            required_before_write_capable_code=self._required_before_write_code(),
            lineage=lineage,
        )

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "real_store_write",
            "write_capable_backend_code",
            "credential_loading",
            "schema_migration_execution",
            "backend_activation",
            "automatic_persistence",
            "permanent_memory_store_mutation",
            "model_weight_mutation",
            "optimizer_mutation",
            "destructive_replacement",
        ]

    @staticmethod
    def _required_before_write_code() -> List[str]:
        return [
            "specific_store_backend_selected",
            "credential_scope_model_reviewed",
            "backup_and_recovery_plan_reviewed",
            "schema_migration_plan_reviewed",
            "destructive_operation_review_completed",
            "dry_run_first_path_implemented",
            "idempotency_and_audit_ledger_tests_defined",
            "explicit_second_authorization_command_received",
        ]


def backend_authorization_contract() -> Dict[str, Any]:
    return {
        "module": "backend_authorization",
        "stage": "FUTURE-BACKEND-AUTHORIZATION",
        "planning_only": True,
        "real_store_write_authorized": False,
        "write_capable_code_authorized": False,
        "requires_second_approval_for_write_capable_code": True,
        "json_safe_decision": True,
    }
