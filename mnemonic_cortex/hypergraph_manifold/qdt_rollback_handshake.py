"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt rollback handshake.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Rollback snapshot handshake requirements for future QDT/WM writes.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .hgm6_result import TransactionCommitPreview, TransactionOperationPreview
from .hgm4_result import TraceSafeMemoryPlan
from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    RollbackSnapshotHandshake,
    RollbackSnapshotRequirement,
    trace_write_prep,
    write_prep_stable_hash,
)
from .qdt_write_contract_probe import _coerce_options
from .validation import ValidationResult


def _operations_from_input(ops_or_preview: Any) -> tuple[TransactionOperationPreview, ...]:
    if isinstance(ops_or_preview, TransactionCommitPreview):
        return tuple(ops_or_preview.operations or tuple())
    if isinstance(ops_or_preview, TransactionOperationPreview):
        return (ops_or_preview,)
    if isinstance(ops_or_preview, TraceSafeMemoryPlan):
        # Handshake can be planned from hooks even before HGM-6 operation previews.
        out = []
        for hook in ops_or_preview.slot_hooks:
            out.append(TransactionOperationPreview(
                operation_id=f"write_prep_op_{write_prep_stable_hash(hook.hook_id, hook.target_slot_id)}",
                operation_type="preview_slot_lattice_write_contract",
                source_payload_id=hook.source_record_id,
                target_slot_id=hook.target_slot_id,
                depth_layer=hook.depth_layer,
                geometry_type=hook.geometry_type,
                qspin_signature_id=hook.qspin_signature_id,
                allowed=False,
                blocked_reason="write-prep generated operation; no HGM-6 commit preview bound",
                trace_id=hook.trace_id,
                metadata={"source_hook_id": hook.hook_id, "write_prep_generated": True},
            ))
        return tuple(out)
    if isinstance(ops_or_preview, Iterable) and not isinstance(ops_or_preview, (str, bytes, Mapping)):
        return tuple(op for op in ops_or_preview if isinstance(op, TransactionOperationPreview))
    return tuple()


def build_rollback_snapshot_handshake(ops_or_preview: Any, config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> RollbackSnapshotHandshake:
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    operations = _operations_from_input(ops_or_preview)
    if not operations:
        validation.warning("rollback_handshake.empty_operations", "no transaction operations supplied", "operations")
    if len(operations) > opts.max_hooks:
        validation.warning("rollback_handshake.bounded_operations", "operation count exceeded max_hooks; requirements truncated", "operations")
    requirements = []
    for op in sorted(operations[: opts.max_hooks], key=lambda o: (o.target_slot_id, o.operation_id)):
        snapshot_ref = f"wm_rollback_snapshot_preview_{write_prep_stable_hash(op.operation_id, op.target_slot_id, length=24)}"
        # This stage cannot bind to actual SystemCommitGate.rollback_stack snapshots.
        bound = False
        blocking = "requires actual WM SystemCommitGate.rollback_stack snapshot binding before write execution"
        validation.warning("rollback_handshake.snapshot_not_bound", blocking, op.operation_id)
        trace = trace_write_prep("qdt_rollback_handshake.requirement", validation, {"operation_id": op.operation_id, "snapshot_ref_preview": snapshot_ref, "bound": bound})
        traces.append(trace)
        requirements.append(RollbackSnapshotRequirement(
            requirement_id=f"rollback_req_{write_prep_stable_hash(op.operation_id, snapshot_ref)}",
            operation_id=op.operation_id,
            target_slot_id=op.target_slot_id,
            snapshot_source="SystemCommitGate.rollback_stack",
            snapshot_ref_preview=snapshot_ref,
            bound_to_actual_snapshot=bound,
            required=True,
            blocking_reason=blocking,
            trace_id=trace.trace_id,
            metadata={"read_only": True, "snapshot_captured": False},
        ))
    complete = bool(requirements and all(req.bound_to_actual_snapshot for req in requirements))
    final_trace = trace_write_prep("qdt_rollback_handshake.build_rollback_snapshot_handshake", validation, {"requirement_count": len(requirements), "complete": complete})
    traces.append(final_trace)
    return RollbackSnapshotHandshake(
        handshake_id=f"rollback_handshake_{write_prep_stable_hash(len(requirements), complete)}",
        requirements=tuple(requirements),
        complete=complete,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"read_only": True, "ready_for_live_write": False},
    )
