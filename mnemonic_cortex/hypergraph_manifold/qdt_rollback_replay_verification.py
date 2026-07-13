"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt rollback replay verification.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Rollback replay verification for HGM/QDT WRITE-PREP-3.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep3_result import (
    HGMQDTWritePrep3Options,
    InMemoryCommitGateSimulationResult,
    RollbackReplayRecord,
    RollbackReplayVerificationResult,
    write_prep3_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_synthetic_slot_store import coerce_write_prep3_options, restore_synthetic_slot_store_from_previous_state
from .validation import ValidationResult


def verify_rollback_replay(
    simulation_result: InMemoryCommitGateSimulationResult,
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> RollbackReplayVerificationResult:
    """Replay rollback inside the synthetic sandbox and verify restoration."""
    opts = coerce_write_prep3_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(simulation_result, InMemoryCommitGateSimulationResult):
        validation.error("prep3.rollback.invalid_simulation", "InMemoryCommitGateSimulationResult is required", "simulation_result")
        trace = trace_write_prep("qdt_rollback_replay_verification.invalid", validation, {"live_rollback_stack_mutated": False})
        return RollbackReplayVerificationResult(
            verification_id=write_prep3_result_id("rollback_replay_invalid", type(simulation_result).__name__),
            replay_records=tuple(),
            verified=False,
            synthetic_store_restored=False,
            live_rollback_stack_mutated=False,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "live_rollback_stack_mutated": False},
        )
    validation.merge(simulation_result.validation)
    restored = restore_synthetic_slot_store_from_previous_state(simulation_result.sandbox_after, config=config, options=opts)
    validation.merge(restored.validation)
    traces.extend(simulation_result.trace_records)
    traces.extend(restored.trace_records)
    before_by_slot = {slot.slot_id: slot.current_fingerprint for slot in simulation_result.sandbox_before.slots}
    restored_by_slot = {slot.slot_id: slot.current_fingerprint for slot in restored.slots}
    records: list[RollbackReplayRecord] = []
    for op in sorted(simulation_result.operations, key=lambda item: (item.target_slot_id, item.operation_id)):
        expected = before_by_slot.get(op.target_slot_id, "")
        actual = restored_by_slot.get(op.target_slot_id, "")
        rollback_applied = bool(op.committed)
        ok = bool((not op.committed) or (expected and expected == actual))
        if op.committed and not ok:
            validation.error("prep3.rollback.restore_mismatch", "rollback replay did not restore expected synthetic state", op.operation_id)
        trace = trace_write_prep("qdt_rollback_replay_verification.record", validation, {
            "operation_id": op.operation_id,
            "rollback_applied": rollback_applied,
            "restored": ok,
            "live_rollback_stack_mutated": False,
        })
        traces.append(trace)
        records.append(RollbackReplayRecord(
            replay_id=write_prep3_result_id("rollback_replay", op.operation_id, expected, actual),
            operation_id=op.operation_id,
            target_slot_id=op.target_slot_id,
            rollback_applied=rollback_applied,
            restored=ok,
            expected_fingerprint=expected,
            actual_fingerprint=actual,
            trace_id=trace.trace_id,
            metadata={"synthetic_only": True, "live_rollback_stack_mutated": False},
        ))
    if not records:
        validation.warning("prep3.rollback.empty_records", "no simulation operations available for rollback replay", "simulation_result.operations")
    verified = bool(records and all(record.restored for record in records))
    final_trace = trace_write_prep("qdt_rollback_replay_verification.verify_rollback_replay", validation, {
        "record_count": len(records),
        "verified": verified,
        "live_rollback_stack_mutated": False,
    })
    traces.append(final_trace)
    return RollbackReplayVerificationResult(
        verification_id=write_prep3_result_id("rollback_replay_verification", simulation_result.simulation_id, verified),
        replay_records=tuple(records),
        verified=verified,
        synthetic_store_restored=verified,
        live_rollback_stack_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-3",
            "synthetic_only": True,
            "live_rollback_stack_mutated": False,
            "verified": verified,
        },
    )
