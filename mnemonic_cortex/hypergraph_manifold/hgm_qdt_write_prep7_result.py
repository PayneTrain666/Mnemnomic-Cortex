"""HGM/QDT WRITE-PREP-7 result contracts.

WRITE-PREP-7 defines permission-token contracts, a shadow commit sandbox, and a
final production-write blocker review.  It remains dry-run/read-only with
respect to live QDT/WM state: no SystemCommitGate.stage/commit, no live
SharedSlotStore write, no live QH write, and no rollback_stack mutation are
performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep7Options:
    """Options for permission-token and shadow commit review.

    ``allow_shadow_commit_sandbox`` allows synthetic shadow commit operations in
    isolated records only. It never opens a live CommitGate boundary.
    """

    max_permission_tokens: int = 128
    max_shadow_operations: int = 128
    max_final_blockers: int = 128
    allow_shadow_commit_sandbox: bool = True
    require_explicit_permission_token: bool = True
    require_human_approval_marker: bool = True
    require_rollback_binding_ready: bool = True
    require_no_live_mutation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("max_permission_tokens", "max_shadow_operations", "max_final_blockers"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PermissionTokenContractRecord:
    token_contract_id: str
    source_binding_id: str
    operation_id: str
    target_slot_id: str
    required_scope: str
    permission_token_id_preview: str
    token_present: bool
    token_validated: bool
    human_approval_marker_present: bool
    write_permission_granted: bool
    can_authorize_live_write: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class PermissionTokenContractResult:
    contract_id: str
    token_records: Tuple[PermissionTokenContractRecord, ...]
    token_contract_ready: bool
    live_write_authorized: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_records", tuple(self.token_records or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ShadowCommitSandboxOperation:
    shadow_operation_id: str
    source_token_contract_id: str
    operation_id: str
    target_slot_id: str
    qh_record_id: str
    rollback_snapshot_ref: str
    would_stage: bool
    would_commit: bool
    shadow_stage_simulated: bool
    shadow_commit_simulated: bool
    live_stage_called: bool
    live_commit_called: bool
    live_store_mutated: bool
    live_qh_mutated: bool
    rollback_stack_mutated: bool
    shadow_success: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class ShadowCommitSandboxResult:
    sandbox_id: str
    operations: Tuple[ShadowCommitSandboxOperation, ...]
    shadow_success: bool
    live_stage_called: bool
    live_commit_called: bool
    live_store_mutated: bool
    live_qh_mutated: bool
    rollback_stack_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class FinalProductionWriteBlocker:
    blocker_id: str
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
class FinalProductionWriteReadinessReview:
    review_id: str
    blockers: Tuple[FinalProductionWriteBlocker, ...]
    open_blocker_count: int
    resolved_blocker_count: int
    high_severity_open_count: int
    shadow_commit_ready: bool
    permission_token_ready: bool
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
class HGMQDTWritePrep7Result:
    permission_token_contract: PermissionTokenContractResult
    shadow_commit_sandbox: ShadowCommitSandboxResult
    final_readiness_review: FinalProductionWriteReadinessReview
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep7_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
