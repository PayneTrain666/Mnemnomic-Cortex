"""QSPIN-PROD-3 commit-gate approval simulation.

This module simulates approval for active-dry-run only. It never performs a
real commit, production activation, payload transfer, QH write, shared-slot
write, or external-memory write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

try:
    from .qspin_production_config import QSpinRuntimeFeatureFlag, QSpinRuntimeKillSwitchState
except Exception:  # pragma: no cover
    from qspin_production_config import QSpinRuntimeFeatureFlag, QSpinRuntimeKillSwitchState


class QSpinCommitGateApprovalSimMode(str, Enum):
    DISABLED = "disabled"
    ACTIVE_DRY_RUN_ONLY = "active_dry_run_only"


class QSpinCommitGateApprovalSimStatus(str, Enum):
    APPROVED_ACTIVE_DRY_RUN = "approved_active_dry_run"
    BLOCKED = "blocked"


class QSpinCommitGateApprovalSimBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    ACTIVE_DRY_RUN_SCOPE_MISSING = "active_dry_run_scope_missing"
    UNSAFE_RUNTIME_FLAG = "unsafe_runtime_flag"
    SOURCE_MATRIX_MISSING = "source_matrix_missing"
    ROLLBACK_EVIDENCE_MISSING = "rollback_evidence_missing"
    KILL_SWITCH_NOT_ENABLED = "kill_switch_not_enabled"
    WRITE_PERMISSION_REQUESTED = "write_permission_requested"
    PAYLOAD_TRANSFER_REQUESTED = "payload_transfer_requested"
    QH_SHARED_SLOT_MUTATION_REQUESTED = "qh_shared_slot_mutation_requested"
    RAW_PAYLOAD_TRACE_REQUESTED = "raw_payload_trace_requested"
    COMMIT_EXECUTION_REQUESTED = "commit_execution_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"


UNSAFE_FLAGS = frozenset({
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
class QSpinCommitGateApprovalPolicy:
    mode: QSpinCommitGateApprovalSimMode = QSpinCommitGateApprovalSimMode.ACTIVE_DRY_RUN_ONLY
    require_active_dry_run_scope: bool = True
    require_source_matrix: bool = True
    require_rollback_evidence: bool = True
    require_kill_switch_enabled: bool = True
    allow_write_permissions: bool = False
    allow_payload_transfer: bool = False
    allow_qh_shared_slot_mutation: bool = False
    allow_raw_payload_trace: bool = False
    allow_commit_execution: bool = False
    allow_production_activation: bool = False

    def validate(self) -> "QSpinCommitGateApprovalPolicy":
        if self.mode is QSpinCommitGateApprovalSimMode.DISABLED:
            raise ValueError("commit-gate approval simulation disabled")
        if not self.require_active_dry_run_scope:
            raise ValueError("active-dry-run scope must be required")
        if not self.require_source_matrix or not self.require_rollback_evidence or not self.require_kill_switch_enabled:
            raise ValueError("source matrix, rollback evidence, and kill-switch must be required")
        if any([
            self.allow_write_permissions,
            self.allow_payload_transfer,
            self.allow_qh_shared_slot_mutation,
            self.allow_raw_payload_trace,
            self.allow_commit_execution,
            self.allow_production_activation,
        ]):
            raise ValueError("unsafe commit-gate approval policy")
        return self


@dataclass(frozen=True)
class QSpinCommitGateApprovalRequest:
    request_id: str
    active_dry_run_scope_declared: bool
    requested_feature_flags: FrozenSet[QSpinRuntimeFeatureFlag] = frozenset()
    source_matrix_complete: bool = False
    rollback_evidence_present: bool = False
    kill_switch_state: QSpinRuntimeKillSwitchState = QSpinRuntimeKillSwitchState.ENABLED
    write_permissions_requested: bool = False
    payload_transfer_requested: bool = False
    qh_shared_slot_mutation_requested: bool = False
    raw_payload_trace_requested: bool = False
    commit_execution_requested: bool = False
    production_activation_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinCommitGateApprovalRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        if not isinstance(self.requested_feature_flags, frozenset):
            raise ValueError("requested_feature_flags must be frozenset")
        for flag in self.requested_feature_flags:
            if not isinstance(flag, QSpinRuntimeFeatureFlag):
                raise ValueError(f"invalid runtime flag: {flag!r}")
        if not isinstance(self.kill_switch_state, QSpinRuntimeKillSwitchState):
            raise ValueError("kill_switch_state must be QSpinRuntimeKillSwitchState")
        return self


@dataclass(frozen=True)
class QSpinCommitGateApprovalDecision:
    status: QSpinCommitGateApprovalSimStatus
    approved_active_dry_run: bool
    block_reasons: Tuple[QSpinCommitGateApprovalSimBlockReason, ...] = ()
    warnings: Tuple[str, ...] = ()

    def validate(self) -> "QSpinCommitGateApprovalDecision":
        if self.status is QSpinCommitGateApprovalSimStatus.APPROVED_ACTIVE_DRY_RUN:
            if not self.approved_active_dry_run or self.block_reasons:
                raise ValueError("approved active-dry-run decision malformed")
        if self.status is QSpinCommitGateApprovalSimStatus.BLOCKED:
            if self.approved_active_dry_run or not self.block_reasons:
                raise ValueError("blocked decision requires reasons and cannot approve")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "approved_active_dry_run": self.approved_active_dry_run,
            "block_reasons": [r.value for r in self.block_reasons],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class QSpinCommitGateApprovalTrace:
    request_id: str
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "safe_summary": dict(self.safe_summary)}


@dataclass(frozen=True)
class QSpinCommitGateApprovalResult:
    request: QSpinCommitGateApprovalRequest
    decision: QSpinCommitGateApprovalDecision
    trace: QSpinCommitGateApprovalTrace
    commit_executed: bool = False
    production_activated: bool = False

    def validate(self) -> "QSpinCommitGateApprovalResult":
        self.request.validate()
        self.decision.validate()
        if self.commit_executed:
            raise ValueError("approval simulation must not execute commits")
        if self.production_activated:
            raise ValueError("approval simulation must not activate production")
        return self


class QSpinCommitGateApprovalSimulator:
    def __init__(self, policy: Optional[QSpinCommitGateApprovalPolicy] = None):
        self.policy = (policy or build_default_qspin_commit_gate_approval_policy()).validate()

    def simulate(self, request: QSpinCommitGateApprovalRequest) -> QSpinCommitGateApprovalResult:
        request.validate()
        reasons = []
        if self.policy.mode is QSpinCommitGateApprovalSimMode.DISABLED:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.MODE_DISABLED)
        if not request.active_dry_run_scope_declared:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.ACTIVE_DRY_RUN_SCOPE_MISSING)
        if request.requested_feature_flags & UNSAFE_FLAGS:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.UNSAFE_RUNTIME_FLAG)
        if not request.source_matrix_complete:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.SOURCE_MATRIX_MISSING)
        if not request.rollback_evidence_present:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.ROLLBACK_EVIDENCE_MISSING)
        if request.kill_switch_state is not QSpinRuntimeKillSwitchState.ENABLED:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.KILL_SWITCH_NOT_ENABLED)
        if request.write_permissions_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.WRITE_PERMISSION_REQUESTED)
        if request.payload_transfer_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.qh_shared_slot_mutation_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.QH_SHARED_SLOT_MUTATION_REQUESTED)
        if request.raw_payload_trace_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.RAW_PAYLOAD_TRACE_REQUESTED)
        if request.commit_execution_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.COMMIT_EXECUTION_REQUESTED)
        if request.production_activation_requested:
            reasons.append(QSpinCommitGateApprovalSimBlockReason.PRODUCTION_ACTIVATION_REQUESTED)

        if reasons:
            decision = QSpinCommitGateApprovalDecision(
                QSpinCommitGateApprovalSimStatus.BLOCKED,
                False,
                tuple(dict.fromkeys(reasons)),
            ).validate()
        else:
            decision = QSpinCommitGateApprovalDecision(
                QSpinCommitGateApprovalSimStatus.APPROVED_ACTIVE_DRY_RUN,
                True,
                (),
                ("approval simulation only; no real commit",),
            ).validate()

        trace = QSpinCommitGateApprovalTrace(
            request.request_id,
            {
                "active_dry_run_scope_declared": request.active_dry_run_scope_declared,
                "source_matrix_complete": request.source_matrix_complete,
                "rollback_evidence_present": request.rollback_evidence_present,
                "kill_switch_state": request.kill_switch_state.value,
                "block_count": len(decision.block_reasons),
            },
        )
        return QSpinCommitGateApprovalResult(request, decision, trace).validate()


def build_default_qspin_commit_gate_approval_policy() -> QSpinCommitGateApprovalPolicy:
    return QSpinCommitGateApprovalPolicy().validate()
