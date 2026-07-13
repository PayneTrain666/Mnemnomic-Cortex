"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-1 commit-gate dry-run inspection hooks.

Dry-run only: no real commit execution, no runtime activation, no payload
transfer, no writes, and no QH/external-memory/shared-slot mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from .qspin_production_config import QSpinRuntimeFeatureFlag, QSpinRuntimeKillSwitchState
except Exception:  # pragma: no cover - standalone loader fallback
    from qspin_production_config import QSpinRuntimeFeatureFlag, QSpinRuntimeKillSwitchState


class QSpinCommitGateDryRunStatus(str, Enum):
    ALLOWED_FOR_SHADOW = "allowed_for_shadow"
    BLOCKED = "blocked"
    WARNING_ONLY = "warning_only"


class QSpinCommitGateDryRunBlockReason(str, Enum):
    UNSAFE_RUNTIME_FLAG = "unsafe_runtime_flag"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"
    SOURCE_MATRIX_MISSING = "source_matrix_missing"
    SOURCE_MATRIX_INCOMPLETE = "source_matrix_incomplete"
    ROLLBACK_EVIDENCE_MISSING = "rollback_evidence_missing"
    KILL_SWITCH_DISABLED = "kill_switch_disabled"
    KILL_SWITCH_TRIPPED = "kill_switch_tripped"
    WRITE_PERMISSION_REQUESTED = "write_permission_requested"
    RAW_PAYLOAD_TRACE_REQUESTED = "raw_payload_trace_requested"
    COMMIT_EXECUTION_REQUESTED = "commit_execution_requested"
    UNKNOWN_UNSAFE_STATE = "unknown_unsafe_state"


UNSAFE_RUNTIME_FLAGS = frozenset({
    QSpinRuntimeFeatureFlag.BRIDGE_ROUTING,
    QSpinRuntimeFeatureFlag.PAYLOAD_TRANSFER,
    QSpinRuntimeFeatureFlag.TOPOLOGY_ROUTING,
    QSpinRuntimeFeatureFlag.DEPTH_PHASE_EXECUTION,
    QSpinRuntimeFeatureFlag.SHARED_SLOT_WRITE,
    QSpinRuntimeFeatureFlag.EXTERNAL_MEMORY_WRITE,
    QSpinRuntimeFeatureFlag.QH_STORAGE_WRITE,
    QSpinRuntimeFeatureFlag.COMMIT_EXECUTION,
    QSpinRuntimeFeatureFlag.RAW_PAYLOAD_TRACE,
    QSpinRuntimeFeatureFlag.PRODUCTION_ACTIVATION,
})


@dataclass(frozen=True)
class QSpinCommitGateDryRunPolicy:
    policy_id: str = "qspin_prod1_commit_gate_dry_run_policy"
    dry_run_only: bool = True
    require_source_matrix: bool = True
    require_rollback_evidence: bool = True
    require_kill_switch_enabled: bool = True
    require_no_write_permissions: bool = True
    require_no_raw_payload_trace: bool = True
    require_no_production_activation: bool = True
    allow_shadow_inspection: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinCommitGateDryRunPolicy":
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.dry_run_only:
            raise ValueError("PROD-1 commit-gate policy must remain dry-run only")
        for name in (
            "require_source_matrix", "require_rollback_evidence", "require_kill_switch_enabled",
            "require_no_write_permissions", "require_no_raw_payload_trace", "require_no_production_activation",
            "allow_shadow_inspection",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be true in PROD-1")
        return self


@dataclass(frozen=True)
class QSpinCommitGateDryRunRequest:
    request_id: str
    requested_feature_flags: FrozenSet[QSpinRuntimeFeatureFlag] = frozenset()
    source_matrix_complete: bool = False
    rollback_evidence_present: bool = False
    kill_switch_state: QSpinRuntimeKillSwitchState = QSpinRuntimeKillSwitchState.ENABLED
    write_permissions_requested: bool = False
    raw_payload_trace_requested: bool = False
    production_activation_requested: bool = False
    commit_execution_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinCommitGateDryRunRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        if not isinstance(self.requested_feature_flags, frozenset):
            raise ValueError("requested_feature_flags must be a frozenset")
        for flag in self.requested_feature_flags:
            if not isinstance(flag, QSpinRuntimeFeatureFlag):
                raise ValueError(f"invalid runtime feature flag: {flag!r}")
        if not isinstance(self.kill_switch_state, QSpinRuntimeKillSwitchState):
            raise ValueError("kill_switch_state must be QSpinRuntimeKillSwitchState")
        return self


@dataclass(frozen=True)
class QSpinCommitGateDryRunDecision:
    status: QSpinCommitGateDryRunStatus
    allowed_for_shadow: bool
    blocked_reasons: Tuple[QSpinCommitGateDryRunBlockReason, ...] = ()
    warnings: Tuple[str, ...] = ()
    required_evidence: Tuple[str, ...] = ()
    rollback_required: bool = True
    kill_switch_required: bool = True
    source_matrix_required: bool = True

    def validate(self) -> "QSpinCommitGateDryRunDecision":
        if self.status is QSpinCommitGateDryRunStatus.ALLOWED_FOR_SHADOW:
            if not self.allowed_for_shadow:
                raise ValueError("allowed status requires allowed_for_shadow=True")
            if self.blocked_reasons:
                raise ValueError("allowed status cannot include blocked reasons")
        if self.status is QSpinCommitGateDryRunStatus.BLOCKED:
            if self.allowed_for_shadow:
                raise ValueError("blocked status must not allow shadow")
            if not self.blocked_reasons:
                raise ValueError("blocked status requires blocked reasons")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed_for_shadow": self.allowed_for_shadow,
            "blocked_reasons": [r.value for r in self.blocked_reasons],
            "warnings": list(self.warnings),
            "required_evidence": list(self.required_evidence),
            "rollback_required": self.rollback_required,
            "kill_switch_required": self.kill_switch_required,
            "source_matrix_required": self.source_matrix_required,
        }


@dataclass(frozen=True)
class QSpinCommitGateDryRunTrace:
    request_id: str
    decision_status: QSpinCommitGateDryRunStatus
    inspected_flags: Tuple[str, ...]
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision_status": self.decision_status.value,
            "inspected_flags": list(self.inspected_flags),
            "safe_summary": dict(self.safe_summary),
        }


@dataclass(frozen=True)
class QSpinCommitGateDryRunResult:
    request: QSpinCommitGateDryRunRequest
    decision: QSpinCommitGateDryRunDecision
    trace: QSpinCommitGateDryRunTrace
    commit_executed: bool = False
    runtime_activated: bool = False

    def validate(self) -> "QSpinCommitGateDryRunResult":
        self.request.validate()
        self.decision.validate()
        if self.commit_executed:
            raise ValueError("PROD-1 dry run must not execute commits")
        if self.runtime_activated:
            raise ValueError("PROD-1 dry run must not activate runtime")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": {
                "request_id": self.request.request_id,
                "requested_feature_flags": sorted(flag.value for flag in self.request.requested_feature_flags),
                "source_matrix_complete": self.request.source_matrix_complete,
                "rollback_evidence_present": self.request.rollback_evidence_present,
                "kill_switch_state": self.request.kill_switch_state.value,
            },
            "decision": self.decision.to_dict(),
            "trace": self.trace.to_dict(),
            "commit_executed": self.commit_executed,
            "runtime_activated": self.runtime_activated,
        }


class QSpinCommitGateDryRunInspector:
    """Deterministic dry-run inspector; performs no real commit."""

    def __init__(self, policy: Optional[QSpinCommitGateDryRunPolicy] = None):
        self.policy = (policy or build_default_qspin_commit_gate_dry_run_policy()).validate()

    def inspect(self, request: QSpinCommitGateDryRunRequest) -> QSpinCommitGateDryRunResult:
        request.validate()
        blocked = []
        required_evidence = []
        warnings = []

        unsafe_flags = sorted((request.requested_feature_flags & UNSAFE_RUNTIME_FLAGS), key=lambda f: f.value)
        if unsafe_flags:
            blocked.append(QSpinCommitGateDryRunBlockReason.UNSAFE_RUNTIME_FLAG)
            warnings.append("unsafe runtime feature flags requested: " + ",".join(flag.value for flag in unsafe_flags))
        if request.production_activation_requested:
            blocked.append(QSpinCommitGateDryRunBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if not request.source_matrix_complete:
            blocked.append(QSpinCommitGateDryRunBlockReason.SOURCE_MATRIX_MISSING)
            required_evidence.append("complete QD6A/QSPIN/PROD source consideration matrix")
        if not request.rollback_evidence_present:
            blocked.append(QSpinCommitGateDryRunBlockReason.ROLLBACK_EVIDENCE_MISSING)
            required_evidence.append("rollback dry-run evidence")
        if request.kill_switch_state is QSpinRuntimeKillSwitchState.DISABLED:
            blocked.append(QSpinCommitGateDryRunBlockReason.KILL_SWITCH_DISABLED)
        if request.kill_switch_state is QSpinRuntimeKillSwitchState.TRIPPED:
            blocked.append(QSpinCommitGateDryRunBlockReason.KILL_SWITCH_TRIPPED)
        if request.write_permissions_requested:
            blocked.append(QSpinCommitGateDryRunBlockReason.WRITE_PERMISSION_REQUESTED)
        if request.raw_payload_trace_requested:
            blocked.append(QSpinCommitGateDryRunBlockReason.RAW_PAYLOAD_TRACE_REQUESTED)
        if request.commit_execution_requested:
            blocked.append(QSpinCommitGateDryRunBlockReason.COMMIT_EXECUTION_REQUESTED)

        if blocked:
            decision = QSpinCommitGateDryRunDecision(
                status=QSpinCommitGateDryRunStatus.BLOCKED,
                allowed_for_shadow=False,
                blocked_reasons=tuple(dict.fromkeys(blocked)),
                warnings=tuple(warnings),
                required_evidence=tuple(dict.fromkeys(required_evidence)),
            )
        else:
            decision = QSpinCommitGateDryRunDecision(
                status=QSpinCommitGateDryRunStatus.ALLOWED_FOR_SHADOW,
                allowed_for_shadow=True,
                warnings=("dry-run only; no commit or runtime activation performed",),
            )
        trace = QSpinCommitGateDryRunTrace(
            request_id=request.request_id,
            decision_status=decision.status,
            inspected_flags=tuple(sorted(flag.value for flag in request.requested_feature_flags)),
            safe_summary={
                "source_matrix_complete": request.source_matrix_complete,
                "rollback_evidence_present": request.rollback_evidence_present,
                "kill_switch_state": request.kill_switch_state.value,
                "blocked_count": len(decision.blocked_reasons),
            },
        )
        return QSpinCommitGateDryRunResult(request, decision.validate(), trace).validate()


def build_default_qspin_commit_gate_dry_run_policy() -> QSpinCommitGateDryRunPolicy:
    return QSpinCommitGateDryRunPolicy(metadata={
        "stage": "QSPIN-PROD-1-QD6A",
        "behavior": "dry-run only; no commit execution",
    }).validate()
