"""QSPIN-PROD-3 QH/shared-slot/external-memory permission dry-run.

Performs metadata-only permission checks. PROD-3 rejects all writes and never
performs real reads/writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class QSpinPermissionDryRunMode(str, Enum):
    DISABLED = "disabled"
    METADATA_CHECK_ONLY = "metadata_check_only"


class QSpinPermissionDryRunStatus(str, Enum):
    APPROVED_SIMULATED_READ = "approved_simulated_read"
    BLOCKED = "blocked"


class QSpinPermissionDryRunBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    UNKNOWN_SCOPE = "unknown_scope"
    UNKNOWN_OPERATION = "unknown_operation"
    WRITE_REJECTED = "write_rejected"
    INTERFERENCE_CHECK_MISSING = "interference_check_missing"
    COMMIT_GATE_REVIEW_MISSING = "commit_gate_review_missing"
    SHARED_SLOT_PERMISSION_METADATA_MISSING = "shared_slot_permission_metadata_missing"
    QH_PERMISSION_METADATA_MISSING = "qh_permission_metadata_missing"
    EXTERNAL_MEMORY_PERMISSION_METADATA_MISSING = "external_memory_permission_metadata_missing"
    RAW_PAYLOAD_PRESENT = "raw_payload_present"


class QSpinPermissionScope(str, Enum):
    SHARED_SLOT = "shared_slot"
    QH = "qh"
    EXTERNAL_MEMORY = "external_memory"


class QSpinPermissionOperation(str, Enum):
    READ_SIMULATION = "read_simulation"
    WRITE = "write"


@dataclass(frozen=True)
class QSpinPermissionDryRunPolicy:
    mode: QSpinPermissionDryRunMode = QSpinPermissionDryRunMode.METADATA_CHECK_ONLY
    allow_real_reads: bool = False
    allow_writes: bool = False
    require_interference_check: bool = True
    require_commit_gate_review: bool = True

    def validate(self) -> "QSpinPermissionDryRunPolicy":
        if self.mode is QSpinPermissionDryRunMode.DISABLED:
            raise ValueError("permission dry-run disabled")
        if self.allow_real_reads:
            raise ValueError("real reads forbidden in PROD-3")
        if self.allow_writes:
            raise ValueError("writes forbidden in PROD-3")
        if not self.require_interference_check or not self.require_commit_gate_review:
            raise ValueError("interference and commit-gate review requirements must stay enabled")
        return self


@dataclass(frozen=True)
class QSpinPermissionDryRunRequest:
    request_id: str
    scope: QSpinPermissionScope
    operation: QSpinPermissionOperation
    target_id: str
    interference_check_present: bool
    commit_gate_review_present: bool
    shared_slot_permission_metadata_present: bool = False
    qh_permission_metadata_present: bool = False
    external_memory_permission_metadata_present: bool = False
    raw_payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinPermissionDryRunRequest":
        if not self.request_id or not self.target_id:
            raise ValueError("request_id and target_id are required")
        if not isinstance(self.scope, QSpinPermissionScope):
            raise ValueError("unknown permission scope")
        if not isinstance(self.operation, QSpinPermissionOperation):
            raise ValueError("unknown permission operation")
        if self.raw_payload is not None:
            raise ValueError("raw payload rejected")
        return self


@dataclass(frozen=True)
class QSpinPermissionDryRunDecision:
    status: QSpinPermissionDryRunStatus
    approved: bool
    block_reasons: Tuple[QSpinPermissionDryRunBlockReason, ...] = ()

    def validate(self) -> "QSpinPermissionDryRunDecision":
        if self.status is QSpinPermissionDryRunStatus.APPROVED_SIMULATED_READ and (not self.approved or self.block_reasons):
            raise ValueError("approved permission dry-run malformed")
        if self.status is QSpinPermissionDryRunStatus.BLOCKED and (self.approved or not self.block_reasons):
            raise ValueError("blocked permission dry-run malformed")
        return self


@dataclass(frozen=True)
class QSpinPermissionDryRunTrace:
    request_id: str
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "safe_summary": dict(self.safe_summary)}


@dataclass(frozen=True)
class QSpinPermissionDryRunResult:
    request: QSpinPermissionDryRunRequest
    decision: QSpinPermissionDryRunDecision
    trace: QSpinPermissionDryRunTrace
    performed_real_read: bool = False
    performed_write: bool = False

    def validate(self) -> "QSpinPermissionDryRunResult":
        self.decision.validate()
        if self.performed_real_read:
            raise ValueError("permission dry-run must not perform real reads")
        if self.performed_write:
            raise ValueError("permission dry-run must not perform writes")
        return self


class QSpinQHSharedSlotPermissionDryRun:
    def __init__(self, policy: Optional[QSpinPermissionDryRunPolicy] = None):
        self.policy = (policy or build_default_qspin_permission_dry_run_policy()).validate()

    def check(self, request: QSpinPermissionDryRunRequest) -> QSpinPermissionDryRunResult:
        reasons = []
        try:
            request.validate()
        except ValueError as exc:
            if "scope" in str(exc):
                reasons.append(QSpinPermissionDryRunBlockReason.UNKNOWN_SCOPE)
            elif "operation" in str(exc):
                reasons.append(QSpinPermissionDryRunBlockReason.UNKNOWN_OPERATION)
            elif "raw payload" in str(exc):
                reasons.append(QSpinPermissionDryRunBlockReason.RAW_PAYLOAD_PRESENT)
            else:
                reasons.append(QSpinPermissionDryRunBlockReason.UNKNOWN_SCOPE)

        if request.operation is QSpinPermissionOperation.WRITE:
            reasons.append(QSpinPermissionDryRunBlockReason.WRITE_REJECTED)
        if not request.interference_check_present:
            reasons.append(QSpinPermissionDryRunBlockReason.INTERFERENCE_CHECK_MISSING)
        if not request.commit_gate_review_present:
            reasons.append(QSpinPermissionDryRunBlockReason.COMMIT_GATE_REVIEW_MISSING)
        if request.scope is QSpinPermissionScope.SHARED_SLOT and not request.shared_slot_permission_metadata_present:
            reasons.append(QSpinPermissionDryRunBlockReason.SHARED_SLOT_PERMISSION_METADATA_MISSING)
        if request.scope is QSpinPermissionScope.QH and not request.qh_permission_metadata_present:
            reasons.append(QSpinPermissionDryRunBlockReason.QH_PERMISSION_METADATA_MISSING)
        if request.scope is QSpinPermissionScope.EXTERNAL_MEMORY and not request.external_memory_permission_metadata_present:
            reasons.append(QSpinPermissionDryRunBlockReason.EXTERNAL_MEMORY_PERMISSION_METADATA_MISSING)

        if reasons:
            decision = QSpinPermissionDryRunDecision(
                QSpinPermissionDryRunStatus.BLOCKED,
                False,
                tuple(dict.fromkeys(reasons)),
            ).validate()
        else:
            decision = QSpinPermissionDryRunDecision(
                QSpinPermissionDryRunStatus.APPROVED_SIMULATED_READ,
                True,
            ).validate()

        trace = QSpinPermissionDryRunTrace(
            request.request_id,
            {
                "scope": request.scope.value if isinstance(request.scope, QSpinPermissionScope) else str(request.scope),
                "operation": request.operation.value if isinstance(request.operation, QSpinPermissionOperation) else str(request.operation),
                "target_id": request.target_id,
                "block_count": len(decision.block_reasons),
            },
        )
        return QSpinPermissionDryRunResult(request, decision, trace).validate()


def build_default_qspin_permission_dry_run_policy() -> QSpinPermissionDryRunPolicy:
    return QSpinPermissionDryRunPolicy().validate()
