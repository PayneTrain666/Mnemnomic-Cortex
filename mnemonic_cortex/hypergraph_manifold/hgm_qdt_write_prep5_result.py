"""HGM/QDT WRITE-PREP-5 result contracts.

WRITE-PREP-5 audits live-shape contract compatibility, permissioned commit
boundaries, and production write blockers.  It remains dry-run/read-only: no
SystemCommitGate.stage/commit, SharedSlotStore write, QH write, rollback_stack
mutation, or production write enablement is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep5Options:
    """Options for live-shape contract and permission-boundary audit.

    ``permissioned_commit_audit`` permits auditing what would be required for a
    future permissioned boundary. It does not permit any live stage/commit call.
    """

    max_shape_checks: int = 128
    max_boundary_checks: int = 128
    max_blockers: int = 128
    permissioned_commit_audit: bool = True
    require_contract_constructed: bool = True
    require_write_permission_false: bool = True
    require_stage_commit_blocked: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_shape_checks) <= 0:
            raise ValueError("max_shape_checks must be positive")
        if int(self.max_boundary_checks) <= 0:
            raise ValueError("max_boundary_checks must be positive")
        if int(self.max_blockers) <= 0:
            raise ValueError("max_blockers must be positive")
        object.__setattr__(self, "max_shape_checks", int(self.max_shape_checks))
        object.__setattr__(self, "max_boundary_checks", int(self.max_boundary_checks))
        object.__setattr__(self, "max_blockers", int(self.max_blockers))


@dataclass(frozen=True)
class LiveShapeContractCheck:
    check_id: str
    preview_id: str
    proposal_id: str
    contract_class_name: str
    constructed: bool
    shape_matches: bool
    content_rank_ok: bool
    finite_content: bool
    confidence_ok: bool
    triplet_ok: bool
    write_permission_false: bool
    live_shape_ready: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class LiveShapeContractHarnessResult:
    harness_id: str
    checks: Tuple[LiveShapeContractCheck, ...]
    live_shape_ready: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class PermissionBoundaryAuditCheck:
    check_id: str
    proposal_id: str
    stage_called: bool
    commit_called: bool
    stage_blocked: bool
    commit_blocked: bool
    write_permission_present: bool
    write_permission_granted: bool
    simulated_permission_only: bool
    permission_boundary_clean: bool
    blocked_reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionedCommitBoundaryAuditResult:
    audit_id: str
    checks: Tuple[PermissionBoundaryAuditCheck, ...]
    permission_boundary_clean: bool
    stage_called: bool
    commit_called: bool
    shared_slot_store_mutated: bool
    qh_storage_mutated: bool
    rollback_stack_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ProductionWriteBlocker:
    blocker_id: str
    source_stage: str
    blocker_code: str
    severity: str
    status: str
    description: str
    required_resolution: str
    evidence: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(str(v) for v in self.evidence or tuple()))


@dataclass(frozen=True)
class ProductionWriteBlockerBurnDownResult:
    register_id: str
    blockers: Tuple[ProductionWriteBlocker, ...]
    open_blocker_count: int
    resolved_blocker_count: int
    high_severity_open_count: int
    production_write_ready: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(self.blockers or tuple()))
        object.__setattr__(self, "open_blocker_count", int(self.open_blocker_count))
        object.__setattr__(self, "resolved_blocker_count", int(self.resolved_blocker_count))
        object.__setattr__(self, "high_severity_open_count", int(self.high_severity_open_count))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrep5Result:
    live_shape_harness: LiveShapeContractHarnessResult
    permission_boundary_audit: PermissionedCommitBoundaryAuditResult
    blocker_burndown: ProductionWriteBlockerBurnDownResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep5_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
