"""HGM/QDT WRITE-PREP-4 result contracts.

WRITE-PREP-4 constructs real QDT/WM contract-object previews in dry-run mode,
defines a synthetic CommitGate adapter boundary, and plans rollback snapshot
binding.  It remains non-mutating: no SystemCommitGate.stage/commit,
SharedSlotStore write, QH storage write, or rollback_stack mutation is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep4Options:
    """Options for real contract-object dry-run construction.

    ``allow_real_contract_construction`` permits local construction of
    SystemWriteProposal objects for validation only. It never permits staging,
    committing, SharedSlotStore writes, QH writes, or rollback_stack mutation.
    """

    max_contract_objects: int = 128
    max_tensor_dim: int = 4096
    allow_real_contract_construction: bool = True
    require_torch_for_real_contracts: bool = True
    allow_adapter_evaluate_preview: bool = True
    bind_synthetic_rollback_snapshots: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_contract_objects) <= 0:
            raise ValueError("max_contract_objects must be positive")
        if int(self.max_tensor_dim) <= 0:
            raise ValueError("max_tensor_dim must be positive")
        object.__setattr__(self, "max_contract_objects", int(self.max_contract_objects))
        object.__setattr__(self, "max_tensor_dim", int(self.max_tensor_dim))


@dataclass(frozen=True)
class RealContractObjectPreview:
    preview_id: str
    source_proposal_id: str
    constructed: bool
    contract_class_name: str
    proposal_id: str
    content_shape: Tuple[int, ...]
    memory_type: str
    local_slot_id: str
    canonical_slot_id: str
    geometry_map: str
    depth_index: int
    triplet_index: int
    bank_name: str
    task_mode: str
    confidence: float
    write_permission: bool
    validation_ready: bool
    object_trace: Mapping[str, Any]
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_shape", tuple(int(v) for v in self.content_shape or tuple()))
        object.__setattr__(self, "depth_index", int(self.depth_index))
        object.__setattr__(self, "triplet_index", int(self.triplet_index))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class RealContractObjectConstructionResult:
    previews: Tuple[RealContractObjectPreview, ...]
    constructed_count: int
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "previews", tuple(self.previews or tuple()))
        object.__setattr__(self, "constructed_count", int(self.constructed_count))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SyntheticCommitGateBoundaryCheck:
    check_id: str
    proposal_preview_id: str
    proposal_id: str
    stage_allowed: bool
    commit_allowed: bool
    evaluate_preview_allowed: bool
    contract_validated: bool
    blocked_reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticCommitGateAdapterBoundary:
    boundary_id: str
    checks: Tuple[SyntheticCommitGateBoundaryCheck, ...]
    stage_called: bool
    commit_called: bool
    shared_slot_store_mutated: bool
    qh_storage_mutated: bool
    rollback_stack_mutated: bool
    evaluation_preview_count: int
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks or tuple()))
        object.__setattr__(self, "evaluation_preview_count", int(self.evaluation_preview_count))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RollbackSnapshotBindingRecord:
    binding_id: str
    requirement_id: str
    operation_id: str
    target_slot_id: str
    snapshot_ref_preview: str
    synthetic_snapshot_ref: str
    actual_rollback_stack_required: bool
    bound_to_actual_snapshot: bool
    binding_ready: bool
    blocked_reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackSnapshotBindingPlan:
    plan_id: str
    bindings: Tuple[RollbackSnapshotBindingRecord, ...]
    complete: bool
    ready_for_live_commit: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrep4Result:
    contract_object_result: RealContractObjectConstructionResult
    adapter_boundary: SyntheticCommitGateAdapterBoundary
    rollback_binding_plan: RollbackSnapshotBindingPlan
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep4_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
