"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm qdt write prep2 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM/QDT WRITE-PREP-2 result contracts.

This module defines dry-run SystemWriteProposal preview records, commit-gate
preflight reports, and non-mutating end-to-end write simulation records.  It is
read-only by design: no class here stages, commits, writes shared slots, writes
QH storage, or mutates rollback stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep2Options:
    """Options for dry-run write proposal simulation.

    ``simulated_write_permission`` is only a dry-run flag used for preflight
    testing. It must never be interpreted as permission to execute QDT/WM writes.
    """

    max_proposals: int = 128
    max_tensor_dim: int = 4096
    simulated_write_permission: bool = False
    allow_torch_validation: bool = True
    require_slot_mapping: bool = True
    require_qh_conversion: bool = True
    require_rollback_handshake: bool = True
    conservative_missing_score: float = 0.25
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_proposals) <= 0:
            raise ValueError("max_proposals must be positive")
        if int(self.max_tensor_dim) <= 0:
            raise ValueError("max_tensor_dim must be positive")
        object.__setattr__(self, "max_proposals", int(self.max_proposals))
        object.__setattr__(self, "max_tensor_dim", int(self.max_tensor_dim))
        object.__setattr__(self, "conservative_missing_score", float(self.conservative_missing_score))


@dataclass(frozen=True)
class DryRunSystemWriteProposalPreview:
    proposal_id: str
    source_preview_id: str
    source_payload_id: str
    content_shape: Tuple[int, ...]
    content_vector: Tuple[float, ...]
    content_fingerprint: str
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
    simulated_write_permission: bool
    tensor_available: bool
    qdt_validation_ready: bool
    ready: bool
    blockers: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_shape", tuple(int(v) for v in self.content_shape or tuple()))
        object.__setattr__(self, "content_vector", tuple(float(v) for v in self.content_vector or tuple()))
        object.__setattr__(self, "depth_index", int(self.depth_index))
        object.__setattr__(self, "triplet_index", int(self.triplet_index))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers or tuple()))


@dataclass(frozen=True)
class DryRunProposalBuilderResult:
    proposals: Tuple[DryRunSystemWriteProposalPreview, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", tuple(self.proposals or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class CommitGatePreflightCheck:
    check_id: str
    proposal_id: str
    check_name: str
    passed: bool
    blocking: bool
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitGatePreflightResult:
    checks: Tuple[CommitGatePreflightCheck, ...]
    ready: bool
    readiness_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks or tuple()))
        object.__setattr__(self, "readiness_score", max(0.0, min(1.0, float(self.readiness_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class EndToEndWriteSimulationReport:
    report_id: str
    proposal_builder_result: DryRunProposalBuilderResult
    preflight_result: CommitGatePreflightResult
    slot_ready: bool
    qh_ready: bool
    rollback_ready: bool
    readiness_score: float
    live_write_executed: bool
    stage_called: bool
    commit_called: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness_score", max(0.0, min(1.0, float(self.readiness_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrep2Result:
    proposal_builder_result: DryRunProposalBuilderResult
    preflight_result: CommitGatePreflightResult
    simulation_report: EndToEndWriteSimulationReport
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep2_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
