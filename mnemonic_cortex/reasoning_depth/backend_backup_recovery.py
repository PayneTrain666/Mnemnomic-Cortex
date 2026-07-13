"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: backend backup recovery.
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


class BackendBackupRecoveryError(ValueError):
    """Raised when backup/recovery planning is unsafe."""


class MigrationDryRunError(ValueError):
    """Raised when migration dry-run planning is unsafe."""


class RecoveryPlanStatus(str, Enum):
    DRAFT_ONLY = "draft_only"
    READY_FOR_REVIEW = "ready_for_review"


@dataclass(frozen=True)
class BackupRecoveryConfig:
    enabled: bool = False
    require_backup_before_write: bool = True
    require_restore_drill: bool = True
    allow_real_backup_execution: bool = False
    allow_real_restore_execution: bool = False
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.allow_real_backup_execution or self.allow_real_restore_execution:
            raise BackendBackupRecoveryError("real backup/restore execution is not allowed in this stage")
        if not self.require_backup_before_write or not self.require_restore_drill:
            raise BackendBackupRecoveryError("backup and restore drill requirements must remain enabled")
        if not self.no_mutation_by_default:
            raise BackendBackupRecoveryError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "BackupRecoveryConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "BackupRecoveryConfig":
        return cls(enabled=True)


@dataclass
class BackupRecoveryPlan:
    enabled: bool
    status: RecoveryPlanStatus
    required_steps: List[str]
    blocked_actions: List[str]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: f"backup_recovery_plan_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "plan_id": self.plan_id,
            "enabled": bool(self.enabled),
            "status": self.status.value,
            "required_steps": list(self.required_steps),
            "blocked_actions": list(self.blocked_actions),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "real_backup_executed": False,
            "real_restore_executed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class BackupRecoveryPlanner:
    def __init__(self, config: Optional[BackupRecoveryConfig] = None):
        self.config = config or BackupRecoveryConfig.disabled()
        self.config.validate()

    def build(self) -> BackupRecoveryPlan:
        if not self.config.enabled:
            return BackupRecoveryPlan(
                enabled=False,
                status=RecoveryPlanStatus.DRAFT_ONLY,
                required_steps=[],
                blocked_actions=self._blocked_actions() + ["backup_recovery_disabled"],
                safety_flags=self._safety_flags(),
            )

        return BackupRecoveryPlan(
            enabled=True,
            status=RecoveryPlanStatus.READY_FOR_REVIEW,
            required_steps=[
                "define_backup_target",
                "define_restore_target",
                "run_dry_run_backup_plan",
                "run_dry_run_restore_plan",
                "verify_restore_hashes",
                "document_recovery_time_objective",
                "document_recovery_point_objective",
            ],
            blocked_actions=self._blocked_actions(),
            safety_flags=self._safety_flags(),
        )

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "real_backup_execution",
            "real_restore_execution",
            "destructive_restore",
            "production_store_mutation",
        ]

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "backup_required_before_write": True,
            "restore_drill_required": True,
            "real_backup_executed": False,
            "real_restore_executed": False,
        }


@dataclass(frozen=True)
class MigrationDryRunConfig:
    enabled: bool = False
    allow_schema_execution: bool = False
    require_dry_run: bool = True
    require_rollback_plan: bool = True
    require_destructive_operation_review: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.allow_schema_execution:
            raise MigrationDryRunError("schema execution is not allowed in this stage")
        if not self.require_dry_run or not self.require_rollback_plan:
            raise MigrationDryRunError("dry-run and rollback plan are required")
        if not self.no_mutation_by_default:
            raise MigrationDryRunError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "MigrationDryRunConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "MigrationDryRunConfig":
        return cls(enabled=True)


@dataclass
class MigrationDryRunPlan:
    enabled: bool
    planned_steps: List[str]
    blocked_actions: List[str]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: f"migration_dry_run_plan_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "plan_id": self.plan_id,
            "enabled": bool(self.enabled),
            "planned_steps": list(self.planned_steps),
            "blocked_actions": list(self.blocked_actions),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "schema_migration_executed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class MigrationDryRunPlanner:
    def __init__(self, config: Optional[MigrationDryRunConfig] = None):
        self.config = config or MigrationDryRunConfig.disabled()
        self.config.validate()

    def build(self) -> MigrationDryRunPlan:
        if not self.config.enabled:
            return MigrationDryRunPlan(
                enabled=False,
                planned_steps=[],
                blocked_actions=self._blocked_actions() + ["migration_dry_run_disabled"],
                safety_flags=self._safety_flags(),
            )

        return MigrationDryRunPlan(
            enabled=True,
            planned_steps=[
                "generate_schema_plan",
                "validate_schema_plan_json",
                "simulate_forward_migration",
                "simulate_rollback_migration",
                "check_destructive_operations",
                "require_second_authorization_before_execution",
            ],
            blocked_actions=self._blocked_actions(),
            safety_flags=self._safety_flags(),
        )

    @staticmethod
    def _blocked_actions() -> List[str]:
        return [
            "schema_migration_execution",
            "destructive_schema_change",
            "production_store_mutation",
            "implicit_schema_upgrade",
        ]

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "schema_migration_executed": False,
            "rollback_plan_required": True,
            "destructive_operation_review_required": True,
        }


def backup_recovery_planning_contract() -> Dict[str, Any]:
    return {
        "module": "backup_recovery_planning",
        "stage": "REAL-BACKEND-IMPLEMENTATION-A",
        "real_backup_executed": False,
        "real_restore_executed": False,
        "backup_required_before_write": True,
        "json_safe_plan": True,
    }


def migration_dry_run_planning_contract() -> Dict[str, Any]:
    return {
        "module": "migration_dry_run_planning",
        "stage": "REAL-BACKEND-IMPLEMENTATION-A",
        "schema_migration_executed": False,
        "dry_run_required": True,
        "rollback_required": True,
        "json_safe_plan": True,
    }
