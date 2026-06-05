"""Rollback snapshot binding plan for WRITE-PREP-4.

This module links WRITE-PREP-1 rollback requirements and WRITE-PREP-3 synthetic
rollback replay evidence into a plan for later real rollback_stack binding.  It
still does not read or mutate a live SystemCommitGate.rollback_stack.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep_result import RollbackSnapshotHandshake, trace_write_prep, write_prep_stable_hash
from .hgm_qdt_write_prep3_result import HGMQDTWritePrep3Result, RollbackReplayVerificationResult
from .hgm_qdt_write_prep4_result import (
    HGMQDTWritePrep4Options,
    RollbackSnapshotBindingPlan,
    RollbackSnapshotBindingRecord,
    write_prep4_result_id,
)
from .qdt_real_contract_object_dryrun import coerce_write_prep4_options
from .validation import ValidationResult


def _extract_replay(obj: Any) -> RollbackReplayVerificationResult | None:
    if isinstance(obj, HGMQDTWritePrep3Result):
        return obj.rollback_replay
    if isinstance(obj, RollbackReplayVerificationResult):
        return obj
    return None


def build_rollback_snapshot_binding_plan(
    rollback_handshake: Any,
    rollback_replay: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep4Options | Mapping[str, Any]] = None,
) -> RollbackSnapshotBindingPlan:
    """Build a dry-run plan for binding previews to future rollback snapshots."""
    opts = coerce_write_prep4_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(rollback_handshake, RollbackSnapshotHandshake):
        validation.error("prep4.rollback_binding.invalid_handshake", "RollbackSnapshotHandshake is required", "rollback_handshake")
        trace = trace_write_prep("qdt_rollback_snapshot_binding_plan.invalid", validation, {"rollback_stack_mutated": False})
        return RollbackSnapshotBindingPlan(
            plan_id=write_prep4_result_id("rollback_binding_invalid", type(rollback_handshake).__name__),
            bindings=tuple(),
            complete=False,
            ready_for_live_commit=False,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "rollback_stack_mutated": False},
        )
    validation.merge(rollback_handshake.validation)
    traces.extend(rollback_handshake.trace_records)
    replay = _extract_replay(rollback_replay)
    replay_by_operation = {}
    if replay is not None:
        validation.merge(replay.validation)
        traces.extend(replay.trace_records)
        replay_by_operation = {record.operation_id: record for record in replay.replay_records}
    elif opts.bind_synthetic_rollback_snapshots:
        validation.warning("prep4.rollback_binding.missing_replay", "no rollback replay evidence supplied; synthetic binding evidence unavailable", "rollback_replay")
    bindings: list[RollbackSnapshotBindingRecord] = []
    for req in sorted(rollback_handshake.requirements, key=lambda item: (item.target_slot_id, item.operation_id)):
        replay_record = replay_by_operation.get(req.operation_id)
        synthetic_ref = f"synthetic_snapshot_{write_prep_stable_hash(req.operation_id, req.target_slot_id, getattr(replay_record, 'expected_fingerprint', 'missing'), length=24)}"
        synthetic_ready = bool(replay_record is not None and replay_record.restored)
        bound_actual = False
        ready = bool(synthetic_ready and bound_actual)
        blocked = "requires live SystemCommitGate.rollback_stack snapshot binding in a later explicit write stage"
        if not synthetic_ready:
            blocked = "synthetic rollback replay evidence missing or not restored"
        validation.warning("prep4.rollback_binding.actual_snapshot_not_bound", blocked, req.operation_id)
        trace = trace_write_prep("qdt_rollback_snapshot_binding_plan.binding", validation, {
            "requirement_id": req.requirement_id,
            "operation_id": req.operation_id,
            "synthetic_ready": synthetic_ready,
            "bound_to_actual_snapshot": bound_actual,
            "rollback_stack_mutated": False,
        })
        traces.append(trace)
        bindings.append(RollbackSnapshotBindingRecord(
            binding_id=write_prep4_result_id("rollback_binding", req.requirement_id, synthetic_ref),
            requirement_id=req.requirement_id,
            operation_id=req.operation_id,
            target_slot_id=req.target_slot_id,
            snapshot_ref_preview=req.snapshot_ref_preview,
            synthetic_snapshot_ref=synthetic_ref,
            actual_rollback_stack_required=True,
            bound_to_actual_snapshot=bound_actual,
            binding_ready=ready,
            blocked_reason=blocked,
            trace_id=trace.trace_id,
            metadata={
                "synthetic_replay_restored": synthetic_ready,
                "actual_snapshot_captured": False,
                "rollback_stack_mutated": False,
            },
        ))
    complete = bool(bindings and all(binding.binding_ready for binding in bindings))
    final_trace = trace_write_prep("qdt_rollback_snapshot_binding_plan.build_rollback_snapshot_binding_plan", validation, {
        "binding_count": len(bindings),
        "complete": complete,
        "ready_for_live_commit": False,
    })
    traces.append(final_trace)
    return RollbackSnapshotBindingPlan(
        plan_id=write_prep4_result_id("rollback_binding_plan", tuple(b.binding_id for b in bindings), complete),
        bindings=tuple(bindings),
        complete=complete,
        ready_for_live_commit=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-4",
            "rollback_stack_mutated": False,
            "actual_snapshot_binding_required": True,
            "ready_for_live_commit": False,
        },
    )
