"""
Plain-language summary
----------------------
What this file is for: Dry-run or sandbox helper (qdt_rollback_snapshot_binding_dryrun).
How it fits in the system: Lets engineers rehearse a path safely without committing live side effects.
Status: LOW-USE / SAFETY SCAFFOLD
Important notes for non-coders: Not the everyday training path.

Technical notes (original):
Rollback snapshot binding dry-run for WRITE-PREP-6.

This module strengthens the WRITE-PREP-4 binding plan by linking it to isolated
SharedSlotStore parity evidence and QHStorageRecord sandbox evidence. It still
does not read or mutate a live SystemCommitGate.rollback_stack.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result, RollbackSnapshotBindingPlan
from .hgm_qdt_write_prep6_result import (
    HGMQDTWritePrep6Options,
    QHStorageRecordSandboxResult,
    RollbackSnapshotBindingDryRunRecord,
    RollbackSnapshotBindingDryRunResult,
    SharedSlotStoreParityHarnessResult,
    write_prep6_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .qdt_real_shared_slot_store_parity import coerce_write_prep6_options
from .validation import ValidationResult


def _plan_from_input(obj: Any) -> RollbackSnapshotBindingPlan | None:
    if isinstance(obj, HGMQDTWritePrep4Result):
        return obj.rollback_binding_plan
    if isinstance(obj, RollbackSnapshotBindingPlan):
        return obj
    return None


def _parity_by_slot(obj: Any) -> dict[str, Any]:
    if isinstance(obj, SharedSlotStoreParityHarnessResult):
        
        out = {}
        for rec in obj.parity_records:
            out[rec.wm_local_slot_id] = rec
            out[rec.observed_canonical_slot_id] = rec
            out[rec.expected_canonical_slot_id] = rec
            if rec.wm_local_slot_id.startswith("wm_"):
                out[rec.wm_local_slot_id[3:]] = rec
                out[rec.wm_local_slot_id[3:].replace("hgm_", "hgm_slot_")] = rec
        return out
    return {}


def _qh_by_slot(obj: Any) -> dict[str, Any]:
    if isinstance(obj, QHStorageRecordSandboxResult):
        return {rec.canonical_slot_id: rec for rec in obj.records}
    return {}


def build_rollback_snapshot_binding_dry_run(
    rollback_plan_or_prep4: Any,
    shared_slot_parity: Any = None,
    qh_sandbox: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep6Options | Mapping[str, Any]] = None,
) -> RollbackSnapshotBindingDryRunResult:
    """Build dry-run rollback snapshot binding records without live stack mutation."""
    opts = coerce_write_prep6_options(options)
    validation = ValidationResult()
    traces = []
    plan = _plan_from_input(rollback_plan_or_prep4)
    if plan is None:
        validation.error("prep6.rollback_dryrun.invalid_plan", "RollbackSnapshotBindingPlan or HGMQDTWritePrep4Result is required", "rollback_plan_or_prep4")
        trace = trace_write_prep("qdt_rollback_snapshot_binding_dryrun.invalid", validation, {"rollback_stack_mutated": False})
        return RollbackSnapshotBindingDryRunResult(
            dry_run_id=write_prep6_result_id("rollback_binding_dryrun_invalid", type(rollback_plan_or_prep4).__name__),
            bindings=tuple(),
            binding_ready=False,
            live_rollback_stack_mutated=False,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "rollback_stack_mutated": False},
        )
    validation.merge(plan.validation)
    traces.extend(plan.trace_records)
    if not opts.allow_rollback_binding_dry_run:
        validation.warning("prep6.rollback_dryrun.disabled", "rollback binding dry-run disabled by options", "options")
    if len(plan.bindings) > opts.max_rollback_bindings:
        validation.warning("prep6.rollback_dryrun.bounded_bindings", "binding count exceeded max_rollback_bindings; records truncated", "bindings")
    parity_map = _parity_by_slot(shared_slot_parity)
    qh_map = _qh_by_slot(qh_sandbox)
    records: list[RollbackSnapshotBindingDryRunRecord] = []
    for binding in sorted(plan.bindings[: opts.max_rollback_bindings], key=lambda b: (b.target_slot_id, b.operation_id)):
        blockers: list[str] = []
        parity = parity_map.get(binding.target_slot_id)
        if parity is None:
            # Target may be local id, while QH evidence is canonical id. Match by any parity canonical if available.
            parity = next((p for p in getattr(shared_slot_parity, "parity_records", tuple()) if p.wm_local_slot_id == binding.target_slot_id or p.expected_canonical_slot_id == binding.target_slot_id), None)
        if parity is None or not getattr(parity, "parity_ok", False):
            blockers.append("SharedSlotStore parity evidence missing or not ok")
        canonical = str(getattr(parity, "observed_canonical_slot_id", "") or getattr(parity, "expected_canonical_slot_id", "") or binding.target_slot_id)
        qh = qh_map.get(canonical)
        if qh is None or not getattr(qh, "validated", False):
            blockers.append("QHStorageRecord sandbox evidence missing or not validated")
        if binding.bound_to_actual_snapshot:
            blockers.append("WRITE-PREP-6 must not bind to actual rollback_stack snapshots")
        if not opts.allow_rollback_binding_dry_run:
            blockers.append("rollback binding dry-run disabled")
        actual_ref = f"rollback_snapshot_preview_{write_prep_stable_hash(binding.operation_id, canonical, getattr(qh, 'qh_record_id', 'missing'), length=24)}"
        ready = bool(not blockers and parity is not None and qh is not None)
        if not ready:
            validation.warning("prep6.rollback_dryrun.not_ready", "; ".join(blockers) or "binding not ready", binding.operation_id)
        trace = trace_write_prep("qdt_rollback_snapshot_binding_dryrun.binding", validation, {
            "operation_id": binding.operation_id,
            "target_slot_id": binding.target_slot_id,
            "binding_ready": ready,
            "rollback_stack_mutated": False,
            "blockers": tuple(blockers),
        })
        traces.append(trace)
        records.append(RollbackSnapshotBindingDryRunRecord(
            binding_id=write_prep6_result_id("rollback_binding_dryrun", binding.binding_id, actual_ref, ready),
            source_binding_id=binding.binding_id,
            operation_id=binding.operation_id,
            target_slot_id=binding.target_slot_id,
            synthetic_snapshot_ref=binding.synthetic_snapshot_ref,
            actual_snapshot_ref_preview=actual_ref,
            parity_evidence_id=str(getattr(parity, "parity_id", "")),
            qh_evidence_id=str(getattr(qh, "sandbox_record_id", "")),
            rollback_stack_required=True,
            rollback_stack_mutated=False,
            binding_ready=ready,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-6",
                "actual_snapshot_preview_only": True,
                "rollback_stack_mutated": False,
                "future_binding_required": True,
            },
        ))
    final_trace = trace_write_prep("qdt_rollback_snapshot_binding_dryrun.build_rollback_snapshot_binding_dry_run", validation, {
        "binding_count": len(records),
        "binding_ready": bool(records and all(r.binding_ready for r in records)),
        "rollback_stack_mutated": False,
    })
    traces.append(final_trace)
    return RollbackSnapshotBindingDryRunResult(
        dry_run_id=write_prep6_result_id("rollback_binding_dryrun", tuple(r.binding_id for r in records)),
        bindings=tuple(records),
        binding_ready=bool(records and all(r.binding_ready for r in records)),
        live_rollback_stack_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-6",
            "rollback_snapshot_binding_preview_only": True,
            "rollback_stack_mutated": False,
            "ready_for_live_commit": False,
        },
    )
