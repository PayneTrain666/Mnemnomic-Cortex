"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm qdt write prep6 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM/QDT WRITE-PREP-6 result contracts.

WRITE-PREP-6 builds isolated real SharedSlotStore parity previews,
QHStorageRecord sandbox construction records, and rollback snapshot binding
previews.  It remains dry-run/read-only with respect to live QDT/WM state:
no live SystemCommitGate.stage/commit, no live SharedSlotStore write, no live QH
storage write, and no rollback_stack mutation are performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep6Options:
    """Options for isolated real-parity and rollback binding dry-run.

    ``allow_isolated_real_shared_slot_store`` permits constructing a brand-new
    in-memory SharedSlotStore instance for parity checks. It never permits
    mutating an existing/live WM store.
    """

    max_parity_records: int = 128
    max_qh_records: int = 128
    max_rollback_bindings: int = 128
    max_tensor_dim: int = 4096
    allow_isolated_real_shared_slot_store: bool = True
    allow_qh_storage_record_sandbox: bool = True
    allow_rollback_binding_dry_run: bool = True
    require_write_permission_false: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("max_parity_records", "max_qh_records", "max_rollback_bindings", "max_tensor_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SharedSlotStoreParityRecord:
    parity_id: str
    source_preview_id: str
    proposal_id: str
    wm_local_slot_id: str
    expected_canonical_slot_id: str
    observed_canonical_slot_id: str
    memory_type: str
    geometry_map: str
    depth_index: int
    vector_fingerprint: str
    isolated_store_constructed: bool
    isolated_write_attempted: bool
    parity_ok: bool
    write_permission_granted: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_index", int(self.depth_index))
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class SharedSlotStoreParityHarnessResult:
    harness_id: str
    parity_records: Tuple[SharedSlotStoreParityRecord, ...]
    parity_ok: bool
    isolated_store_mutated: bool
    live_store_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parity_records", tuple(self.parity_records or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class QHStorageRecordSandboxRecord:
    sandbox_record_id: str
    source_preview_id: str
    proposal_id: str
    canonical_slot_id: str
    qh_record_id: str
    composite_code: str
    vector_fingerprint: str
    vector_norm: float
    depth_index: int
    geometry_map: str
    triplet_index: int
    memory_type: str
    task_mode: str
    confidence: float
    write_permission_granted: bool
    constructed: bool
    validated: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector_norm", float(self.vector_norm))
        object.__setattr__(self, "depth_index", int(self.depth_index))
        object.__setattr__(self, "triplet_index", int(self.triplet_index))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class QHStorageRecordSandboxResult:
    sandbox_id: str
    records: Tuple[QHStorageRecordSandboxRecord, ...]
    constructed_count: int
    validated_count: int
    sandbox_qh_storage_mutated: bool
    live_qh_storage_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or tuple()))
        object.__setattr__(self, "constructed_count", int(self.constructed_count))
        object.__setattr__(self, "validated_count", int(self.validated_count))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RollbackSnapshotBindingDryRunRecord:
    binding_id: str
    source_binding_id: str
    operation_id: str
    target_slot_id: str
    synthetic_snapshot_ref: str
    actual_snapshot_ref_preview: str
    parity_evidence_id: str
    qh_evidence_id: str
    rollback_stack_required: bool
    rollback_stack_mutated: bool
    binding_ready: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class RollbackSnapshotBindingDryRunResult:
    dry_run_id: str
    bindings: Tuple[RollbackSnapshotBindingDryRunRecord, ...]
    binding_ready: bool
    live_rollback_stack_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrep6Result:
    shared_slot_parity: SharedSlotStoreParityHarnessResult
    qh_sandbox: QHStorageRecordSandboxResult
    rollback_binding_dry_run: RollbackSnapshotBindingDryRunResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep6_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
