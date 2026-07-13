"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm6 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-6 result dataclasses for write-permission gates and commit previews.

HGM-6 remains preview-only. It builds transactional operation previews,
rollback manifests, and commit-readiness scores without writing into QDT/WM
or mutating any live memory internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .enums import DepthLayer, GeometryType
from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class HGM6CommitOptions:
    """Options for preview-only transaction planning.

    Defaults are intentionally conservative: no write request, no write grant,
    dry-run enabled, preview-only enabled, and commit preview not allowed.
    """

    requested: bool = False
    granted: bool = False
    dry_run: bool = True
    preview_only: bool = True
    allow_commit_preview: bool = False
    max_operations: int = 128
    readiness_threshold: float = 0.75
    conservative_missing_hgm5_score: float = 0.35
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_operations) <= 0:
            raise ValueError("max_operations must be positive")
        object.__setattr__(self, "max_operations", int(self.max_operations))
        object.__setattr__(self, "readiness_threshold", max(0.0, min(1.0, float(self.readiness_threshold))))
        object.__setattr__(self, "conservative_missing_hgm5_score", max(0.0, min(1.0, float(self.conservative_missing_hgm5_score))))


@dataclass(frozen=True)
class WritePermissionState:
    permission_id: str
    requested: bool
    granted: bool
    reason: str
    dry_run: bool
    preview_only: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionOperationPreview:
    operation_id: str
    operation_type: str
    source_payload_id: str
    target_slot_id: str
    depth_layer: DepthLayer
    geometry_type: GeometryType
    qspin_signature_id: str
    allowed: bool
    blocked_reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))
        object.__setattr__(self, "geometry_type", GeometryType.coerce(self.geometry_type))


@dataclass(frozen=True)
class RollbackOperation:
    rollback_id: str
    operation_id: str
    rollback_type: str
    target_slot_id: str
    previous_state_ref: str
    allowed: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackManifest:
    manifest_id: str
    operations: Tuple[RollbackOperation, ...]
    complete: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class CommitReadinessScore:
    score_id: str
    score: float
    confidence: float
    ready: bool
    blockers: Tuple[str, ...]
    warnings: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "blockers", tuple(self.blockers or tuple()))
        object.__setattr__(self, "warnings", tuple(self.warnings or tuple()))


@dataclass(frozen=True)
class TransactionCommitPreview:
    preview_id: str
    operations: Tuple[TransactionOperationPreview, ...]
    rollback_manifest: RollbackManifest
    write_permission: WritePermissionState
    commit_readiness: CommitReadinessScore
    allowed: bool
    blocked_reason: str
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM6WritePermissionResult:
    write_permission: WritePermissionState
    transaction_preview: TransactionCommitPreview
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
