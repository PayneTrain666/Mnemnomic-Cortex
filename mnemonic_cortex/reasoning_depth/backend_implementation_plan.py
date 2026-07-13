"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: backend implementation plan.
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


class BackendImplementationPlanError(ValueError):
    """Raised when future backend implementation plan is unsafe."""


class BackendPlanStatus(str, Enum):
    DRAFT_PLAN_ONLY = "draft_plan_only"
    BLOCKED_PENDING_REVIEW = "blocked_pending_review"
    READY_FOR_SEPARATE_AUTHORIZATION = "ready_for_separate_authorization"


@dataclass(frozen=True)
class BackendImplementationPlanConfig:
    enabled: bool = False
    include_sqlite_option: bool = True
    include_jsonl_option: bool = True
    include_postgres_option: bool = True
    include_vector_store_option: bool = True
    require_dry_run_first: bool = True
    require_idempotency: bool = True
    require_backup_plan: bool = True
    require_rollback_plan: bool = True
    require_redaction_gate: bool = True
    allow_write_capable_code: bool = False
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.allow_write_capable_code:
            raise BackendImplementationPlanError("write-capable backend code is not allowed in planning stage")
        if not self.require_dry_run_first or not self.require_idempotency:
            raise BackendImplementationPlanError("dry-run-first and idempotency are required")
        if not self.no_mutation_by_default:
            raise BackendImplementationPlanError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "BackendImplementationPlanConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "BackendImplementationPlanConfig":
        return cls(enabled=True)


@dataclass
class BackendImplementationOption:
    option_id: str
    backend_kind: str
    strengths: List[str]
    risks: List[str]
    required_controls: List[str]
    recommended_first: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "option_id": self.option_id,
            "backend_kind": self.backend_kind,
            "strengths": list(self.strengths),
            "risks": list(self.risks),
            "required_controls": list(self.required_controls),
            "recommended_first": bool(self.recommended_first),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class BackendImplementationPlanReport:
    enabled: bool
    status: BackendPlanStatus
    options: List[BackendImplementationOption]
    implementation_phases: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"backend_implementation_plan_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "status": self.status.value,
            "options": [option.to_dict() for option in self.options],
            "implementation_phases": list(self.implementation_phases),
            "acceptance_tests": list(self.acceptance_tests),
            "blocked_actions": list(self.blocked_actions),
            "lineage": _safe_jsonable(self.lineage),
            "write_capable_code_generated": False,
            "real_store_write_authorized": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class BackendImplementationPlanner:
    """Builds a future backend implementation plan without write-capable code."""

    def __init__(self, config: Optional[BackendImplementationPlanConfig] = None):
        self.config = config or BackendImplementationPlanConfig.disabled()
        self.config.validate()

    def build(self, *, lineage: Optional[Dict[str, Any]] = None) -> BackendImplementationPlanReport:
        if not self.config.enabled:
            return BackendImplementationPlanReport(
                enabled=False,
                status=BackendPlanStatus.BLOCKED_PENDING_REVIEW,
                options=[],
                blocked_actions=self._blocked_actions() + ["plan_disabled"],
                lineage=lineage or {},
            )

        options: List[BackendImplementationOption] = []
        if self.config.include_jsonl_option:
            options.append(self._option("jsonl", ["simple append-only audit", "easy inspection"], ["poor concurrent writes", "manual compaction"], recommended=True))
        if self.config.include_sqlite_option:
            options.append(self._option("sqlite", ["transactional local store", "low operational burden"], ["single-writer limits", "file-lock semantics"], recommended=True))
        if self.config.include_postgres_option:
            options.append(self._option("postgres", ["strong transactions", "multi-client scale"], ["credential and ops burden", "migration risk"]))
        if self.config.include_vector_store_option:
            options.append(self._option("vector_store", ["retrieval-friendly embeddings", "semantic lookup"], ["embedding drift", "privacy/redaction complexity"]))

        return BackendImplementationPlanReport(
            enabled=True,
            status=BackendPlanStatus.READY_FOR_SEPARATE_AUTHORIZATION,
            options=options,
            implementation_phases=[
                "phase_0_design_review_only",
                "phase_1_dry_run_backend_interface",
                "phase_2_local_shadow_write_tests",
                "phase_3_backup_restore_drill",
                "phase_4_limited_write_capable_code_after_explicit_authorization",
                "phase_5_disabled_default_release",
            ],
            acceptance_tests=[
                "no_write_without_second_authorization",
                "dry_run_first_default",
                "idempotency_duplicate_rejection",
                "redaction_gate_blocks_sensitive_payloads",
                "backup_restore_drill_required",
                "migration_dry_run_required",
                "permission_scope_denial_tests",
                "rollback_metadata_round_trip",
            ],
            blocked_actions=self._blocked_actions(),
            lineage=lineage or {},
        )

    @staticmethod
    def _option(kind: str, strengths: List[str], risks: List[str], recommended: bool = False) -> BackendImplementationOption:
        controls = [
            "dry_run_first",
            "idempotency_key_required",
            "audit_ledger_required",
            "redaction_gate_required",
            "backup_restore_required",
            "write_permission_required",
            "disabled_by_default",
        ]
        return BackendImplementationOption(
            option_id=f"backend_option_{kind}_{uuid.uuid4().hex[:8]}",
            backend_kind=kind,
            strengths=strengths,
            risks=risks,
            required_controls=controls,
            recommended_first=recommended,
        )

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "write_capable_code_generation",
            "credential_loading",
            "schema_migration_execution",
            "real_store_write",
            "backend_activation",
            "automatic_persistence",
        ]


def backend_implementation_plan_contract() -> Dict[str, Any]:
    return {
        "module": "backend_implementation_plan",
        "stage": "FUTURE-BACKEND-AUTHORIZATION",
        "planning_only": True,
        "write_capable_code_generated": False,
        "real_store_write_authorized": False,
        "separate_authorization_required": True,
        "json_safe_report": True,
    }
