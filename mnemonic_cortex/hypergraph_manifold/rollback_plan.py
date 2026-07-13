"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: rollback plan.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Rollback manifest construction for HGM-6 commit previews.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional, Sequence

from .hgm6_result import HGM6CommitOptions, RollbackManifest, RollbackOperation, TransactionOperationPreview
from .validation import ValidationResult
from .write_permission_gate import coerce_hgm6_options, trace_hgm6


def _stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def build_rollback_manifest(
    operation_previews: Sequence[TransactionOperationPreview],
    config=None,
    options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None,
) -> RollbackManifest:
    """Build preview rollback operations for each transaction operation."""

    validation = ValidationResult()
    traces: List[Any] = []
    ops = tuple(operation_previews or tuple())
    if not ops:
        validation.warning("hgm6_rollback.empty_operations", "no operation previews supplied; empty rollback manifest", "operation_previews")
    rollback_ops: List[RollbackOperation] = []
    for op in sorted(ops, key=lambda item: (item.target_slot_id, item.operation_id)):
        previous_state_ref = f"preview_previous_state::{op.target_slot_id}::{_stable_hash(op.operation_id)}"
        trace = trace_hgm6("rollback_plan.rollback_operation", validation, {"operation_id": op.operation_id, "target_slot_id": op.target_slot_id})
        traces.append(trace)
        rollback_ops.append(RollbackOperation(
            rollback_id=f"hgm6_rb_{_stable_hash(op.operation_id, op.target_slot_id)}",
            operation_id=op.operation_id,
            rollback_type="preview_restore_slot_state_ref",
            target_slot_id=op.target_slot_id,
            previous_state_ref=previous_state_ref,
            allowed=bool(op.allowed),
            trace_id=trace.trace_id,
            metadata={"preview_only": True, "executed": False},
        ))
    allowed_ops = tuple(op for op in ops if op.allowed)
    covered = {rb.operation_id for rb in rollback_ops if rb.previous_state_ref}
    complete = all(op.operation_id in covered for op in allowed_ops)
    if allowed_ops and not complete:
        validation.error("hgm6_rollback.incomplete", "rollback manifest missing coverage for allowed operations", "operations")
    if not allowed_ops:
        validation.warning("hgm6_rollback.no_allowed_operations", "no allowed operation previews require rollback coverage", "operations")
    final_trace = trace_hgm6("rollback_plan.build_rollback_manifest", validation, {"rollback_count": len(rollback_ops), "complete": complete})
    traces.append(final_trace)
    return RollbackManifest(
        manifest_id=f"hgm6_rb_manifest_{_stable_hash(tuple(rb.rollback_id for rb in rollback_ops), complete)}",
        operations=tuple(rollback_ops),
        complete=bool(complete),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"rollback_count": len(rollback_ops), "allowed_operation_count": len(allowed_ops), "preview_only": True},
    )
