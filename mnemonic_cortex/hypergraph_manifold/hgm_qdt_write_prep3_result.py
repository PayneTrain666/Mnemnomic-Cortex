"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm qdt write prep3 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM/QDT WRITE-PREP-3 result contracts.

WRITE-PREP-3 adds an isolated in-memory CommitGate simulation, a synthetic
SharedSlotStore-like sandbox, and rollback replay verification.  These contracts
remain non-mutating with respect to real QDT/WM internals: no SystemCommitGate,
SharedSlotStore, QH storage, or rollback_stack object is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult
from .hgm_qdt_write_prep_result import write_prep_stable_hash


@dataclass(frozen=True)
class HGMQDTWritePrep3Options:
    """Options for isolated in-memory write simulation.

    ``allow_synthetic_commit`` only affects the local synthetic sandbox. It must
    never be interpreted as permission to mutate QDT/WM state.
    """

    max_operations: int = 128
    max_slot_payload_values: int = 4096
    allow_synthetic_commit: bool = True
    allow_preflight_blocked_simulation: bool = True
    require_proposal_ready: bool = True
    verify_rollback_after_simulation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_operations) <= 0:
            raise ValueError("max_operations must be positive")
        if int(self.max_slot_payload_values) <= 0:
            raise ValueError("max_slot_payload_values must be positive")
        object.__setattr__(self, "max_operations", int(self.max_operations))
        object.__setattr__(self, "max_slot_payload_values", int(self.max_slot_payload_values))


@dataclass(frozen=True)
class SyntheticSlotRecord:
    slot_id: str
    canonical_slot_id: str
    current_vector: Tuple[float, ...]
    current_fingerprint: str
    previous_vector: Tuple[float, ...]
    previous_fingerprint: str
    version: int
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_vector", tuple(float(v) for v in self.current_vector or tuple()))
        object.__setattr__(self, "previous_vector", tuple(float(v) for v in self.previous_vector or tuple()))
        object.__setattr__(self, "version", int(self.version))


@dataclass(frozen=True)
class SyntheticSharedSlotStoreSandbox:
    sandbox_id: str
    slots: Tuple[SyntheticSlotRecord, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class InMemoryCommitGateSimulationOperation:
    operation_id: str
    proposal_id: str
    target_slot_id: str
    canonical_slot_id: str
    staged: bool
    committed: bool
    blocked_reason: str
    before_fingerprint: str
    after_fingerprint: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InMemoryCommitGateSimulationResult:
    simulation_id: str
    operations: Tuple[InMemoryCommitGateSimulationOperation, ...]
    sandbox_before: SyntheticSharedSlotStoreSandbox
    sandbox_after: SyntheticSharedSlotStoreSandbox
    preflight_ready: bool
    synthetic_store_mutated: bool
    live_store_mutated: bool
    stage_called: bool
    commit_called: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RollbackReplayRecord:
    replay_id: str
    operation_id: str
    target_slot_id: str
    rollback_applied: bool
    restored: bool
    expected_fingerprint: str
    actual_fingerprint: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackReplayVerificationResult:
    verification_id: str
    replay_records: Tuple[RollbackReplayRecord, ...]
    verified: bool
    synthetic_store_restored: bool
    live_rollback_stack_mutated: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "replay_records", tuple(self.replay_records or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrep3Result:
    commitgate_simulation: InMemoryCommitGateSimulationResult
    rollback_replay: RollbackReplayVerificationResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


def write_prep3_result_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{write_prep_stable_hash(*parts)}"
