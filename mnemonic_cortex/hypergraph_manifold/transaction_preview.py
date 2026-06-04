"""Preview-only transaction operation generation for HGM-6."""

from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional, Tuple

from .hgm4_result import TraceSafeMemoryPlan
from .hgm6_result import HGM6CommitOptions, TransactionOperationPreview
from .validation import ValidationResult
from .write_permission_gate import coerce_hgm6_options, trace_hgm6


def _stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def build_transaction_operation_previews(
    memory_plan: TraceSafeMemoryPlan,
    config=None,
    options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None,
) -> Tuple[Tuple[TransactionOperationPreview, ...], ValidationResult, Tuple[Any, ...]]:
    """Convert HGM-4 slot hooks into deterministic transaction previews.

    No write is ever executed. The operation ``allowed`` flag only means the
    operation is structurally previewable.
    """

    opts = coerce_hgm6_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    if not isinstance(memory_plan, TraceSafeMemoryPlan):
        validation.error("hgm6_ops.invalid_memory_plan", "memory_plan must be a TraceSafeMemoryPlan", "memory_plan")
        trace = trace_hgm6("transaction_preview.build_transaction_operation_previews", validation, {"reason": "invalid_memory_plan"})
        return tuple(), validation, (trace,)
    hooks = tuple(memory_plan.slot_hooks or tuple())
    if not hooks:
        validation.warning("hgm6_ops.empty_hooks", "memory plan contains no slot hooks; empty transaction preview", "slot_hooks")
    if len(hooks) > opts.max_operations:
        validation.warning("hgm6_ops.bounded_operation_count", "slot hook count exceeded max_operations; operations truncated", "slot_hooks")
    operations: List[TransactionOperationPreview] = []
    sorted_hooks = sorted(hooks[: opts.max_operations], key=lambda h: (h.depth_layer.value, h.target_slot_id, h.source_record_id, h.hook_id))
    for hook in sorted_hooks:
        blocked = ""
        allowed = True
        if not memory_plan.dry_run or not hook.dry_run:
            allowed = False
            blocked = "operation is not dry-run safe"
            validation.warning("hgm6_ops.not_dry_run", blocked, hook.hook_id)
        if memory_plan.write_intent or hook.write_intent:
            allowed = False
            blocked = "write_intent present; HGM-6 operations remain preview-only"
            validation.warning("hgm6_ops.write_intent_preview_only", blocked, hook.hook_id)
        if not hook.target_slot_id:
            allowed = False
            blocked = "missing target slot id"
            validation.error("hgm6_ops.missing_target_slot", blocked, hook.hook_id)
        op_trace = trace_hgm6("transaction_preview.operation", validation, {"hook_id": hook.hook_id, "allowed": allowed, "blocked_reason": blocked})
        traces.append(op_trace)
        operations.append(TransactionOperationPreview(
            operation_id=f"hgm6_op_{_stable_hash(hook.hook_id, hook.source_record_id, hook.target_slot_id)}",
            operation_type="preview_slot_lattice_write_contract",
            source_payload_id=hook.source_record_id,
            target_slot_id=hook.target_slot_id,
            depth_layer=hook.depth_layer,
            geometry_type=hook.geometry_type,
            qspin_signature_id=hook.qspin_signature_id,
            allowed=allowed,
            blocked_reason=blocked,
            trace_id=op_trace.trace_id,
            metadata={"hook_id": hook.hook_id, "executed": False, "preview_only": True},
        ))
    final_trace = trace_hgm6("transaction_preview.build_transaction_operation_previews", validation, {"operation_count": len(operations), "max_operations": opts.max_operations})
    traces.append(final_trace)
    return tuple(operations), validation, tuple(traces)

from .commit_readiness import score_commit_readiness
from .hgm6_result import TransactionCommitPreview
from .rollback_plan import build_rollback_manifest
from .write_permission_gate import build_write_permission_state


def _empty_rollback_manifest(validation: ValidationResult):
    from .hgm6_result import RollbackManifest
    trace = trace_hgm6("transaction_preview.empty_rollback_manifest", validation, {"empty": True})
    return RollbackManifest("hgm6_rb_manifest_invalid", tuple(), False, validation, (trace,), metadata={"empty": True, "preview_only": True})


def build_transaction_commit_preview(
    memory_plan: TraceSafeMemoryPlan,
    hgm5_result=None,
    config=None,
    options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None,
) -> TransactionCommitPreview:
    """Build a full preview-only transaction commit plan.

    The returned preview never executes writes. ``allowed`` means the preview
    has cleared readiness checks for a future explicit write-execution stage.
    """

    opts = coerce_hgm6_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    permission = build_write_permission_state(requested=opts.requested, granted=opts.granted, config=config, options=opts)
    ops, op_validation, op_traces = build_transaction_operation_previews(memory_plan, config=config, options=opts)
    validation.merge(op_validation)
    traces.extend(op_traces)
    rollback = build_rollback_manifest(ops, config=config, options=opts)
    validation.merge(rollback.validation)
    traces.extend(rollback.trace_records)
    readiness = score_commit_readiness(memory_plan, hgm5_result=hgm5_result, rollback_manifest=rollback, write_permission=permission, config=config, options=opts)
    # Convert readiness warnings/blockers into preview validation without making warnings fatal.
    if readiness.blockers:
        validation.warning("hgm6_preview.readiness_blocked", "; ".join(readiness.blockers), "commit_readiness")
    allowed = bool(readiness.ready and permission.granted and opts.allow_commit_preview and opts.preview_only and opts.dry_run)
    blocked_reason = ""
    if not allowed:
        if not opts.allow_commit_preview:
            blocked_reason = "explicit commit preview permission not enabled"
        elif not permission.granted:
            blocked_reason = "write permission not granted"
        elif not readiness.ready:
            blocked_reason = "; ".join(readiness.blockers) or "commit readiness checks did not pass"
        else:
            blocked_reason = "preview-only/dry-run safety settings block live commit"
    trace = trace_hgm6("transaction_preview.build_transaction_commit_preview", validation, {"allowed": allowed, "blocked_reason": blocked_reason, "operation_count": len(ops), "executed": False})
    traces.append(trace)
    return TransactionCommitPreview(
        preview_id=f"hgm6_commit_preview_{_stable_hash(getattr(memory_plan, 'plan_id', 'invalid'), permission.permission_id, readiness.score_id, allowed)}",
        operations=ops,
        rollback_manifest=rollback,
        write_permission=permission,
        commit_readiness=readiness,
        allowed=allowed,
        blocked_reason=blocked_reason,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"preview_only": True, "executed": False, "operation_count": len(ops), "allow_commit_preview": opts.allow_commit_preview},
    )
