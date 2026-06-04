"""Trace-safe HGM-4 memory integration plans.

Plans and previews are read-only contracts. This module never writes to the
working-memory/QDT runtime and never calls hardware or network services.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional, Sequence

from .bridge_payloads import build_bridge_payload_from_hgm_record
from .enums import TraceEventKind, ValidationSeverity
from .hgm4_result import BridgeExecutionPreview, HGMBridgePayload, QDTWMBridgeOptions, TraceSafeMemoryPlan
from .qdt_wm_bridge import detect_qdt_wm_adapter_status
from .slot_lattice_hooks import build_shared_slot_lattice_hooks
from .types import TraceRecord
from .validation import ValidationResult


def _stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _coerce_options(options: Optional[QDTWMBridgeOptions | Mapping[str, Any]]) -> QDTWMBridgeOptions:
    if options is None:
        return QDTWMBridgeOptions()
    if isinstance(options, QDTWMBridgeOptions):
        return options
    return QDTWMBridgeOptions(**dict(options))


def _trace(component: str, validation: ValidationResult, payload=None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload=dict(payload or {}),
    )


def build_trace_safe_memory_plan(records: Sequence[Any], config=None, options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None) -> TraceSafeMemoryPlan:
    """Build a dry-run trace-safe memory plan from HGM records."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    adapter = detect_qdt_wm_adapter_status(config=config, options=opts)
    if not adapter.available:
        validation.warning("hgm4_plan.adapter_unavailable", adapter.reason, "adapter_status")
    recs = tuple(records or tuple())
    if not recs:
        validation.warning("hgm4_plan.empty_records", "no HGM records supplied; empty bridge plan generated", "records")
    if len(recs) > opts.max_payload_count:
        validation.warning("hgm4_plan.bounded_payload_count", "records exceeded max_payload_count; payloads were truncated", "records")

    payloads: List[HGMBridgePayload] = []
    for idx, record in enumerate(recs[: opts.max_payload_count]):
        built = build_bridge_payload_from_hgm_record(record, config=config, options=opts)
        validation.merge(built.validation)
        traces.extend(built.trace_records)
        if built.payload is not None:
            payloads.append(built.payload)

    payloads.sort(key=lambda p: (p.depth_layer.value, p.source_type, p.source_id, p.payload_id))
    hooks = build_shared_slot_lattice_hooks(payloads, config=config, options=opts)
    validation.merge(hooks.validation)
    traces.extend(hooks.trace_records)

    plan_trace = _trace(
        "trace_safe_memory_integration.build_trace_safe_memory_plan",
        validation,
        {"payload_count": len(payloads), "hook_count": len(hooks.hooks), "dry_run": opts.dry_run, "write_intent": opts.write_intent},
    )
    traces.append(plan_trace)
    return TraceSafeMemoryPlan(
        plan_id=f"hgm4_plan_{_stable_hash(tuple(p.payload_id for p in payloads), tuple(h.hook_id for h in hooks.hooks), opts.dry_run, opts.write_intent)}",
        bridge_payloads=tuple(payloads),
        slot_hooks=hooks.hooks,
        adapter_status=adapter,
        dry_run=bool(opts.dry_run),
        write_intent=bool(opts.write_intent),
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "payload_count": len(payloads),
            "hook_count": len(hooks.hooks),
            "adapter_available": adapter.available,
            "preview_only": True,
        },
    )


def preview_bridge_execution(plan: TraceSafeMemoryPlan, config=None, options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None) -> BridgeExecutionPreview:
    """Create a preview-only execution plan without performing writes."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    if not isinstance(plan, TraceSafeMemoryPlan):
        validation.error("hgm4_preview.invalid_plan", "plan must be a TraceSafeMemoryPlan", "plan")
        trace = _trace("trace_safe_memory_integration.preview_bridge_execution", validation, {"reason": "invalid_plan"})
        return BridgeExecutionPreview("hgm4_preview_invalid", "", False, "invalid plan", tuple(), validation, (trace,), metadata={"preview_only": True})

    blocked_reason = ""
    allowed = True
    if plan.write_intent and not opts.allow_write_preview:
        allowed = False
        blocked_reason = "write_intent=True is blocked unless allow_write_preview=True; no writes were executed"
        validation.warning("hgm4_preview.write_intent_blocked", blocked_reason, "write_intent")
    elif not plan.adapter_status.available:
        # Still allow a structural preview; no write can occur anyway.
        validation.warning("hgm4_preview.adapter_unavailable", "adapter unavailable; structural preview only", "adapter_status")

    operations = []
    for hook in plan.slot_hooks:
        operations.append(
            f"preview_bridge_payload source={hook.source_record_id} target_slot={hook.target_slot_id} depth={hook.depth_layer.name} geometry={hook.geometry_type.value} dry_run={hook.dry_run} write_intent={hook.write_intent}"
        )
    if not operations:
        validation.warning("hgm4_preview.empty_operations", "no bridge operations to preview", "planned_operations")
    trace = _trace(
        "trace_safe_memory_integration.preview_bridge_execution",
        validation,
        {"plan_id": plan.plan_id, "allowed": allowed, "operation_count": len(operations), "blocked_reason": blocked_reason},
    )
    traces.append(trace)
    return BridgeExecutionPreview(
        preview_id=f"hgm4_preview_{_stable_hash(plan.plan_id, allowed, len(operations), blocked_reason)}",
        plan_id=plan.plan_id,
        allowed=allowed,
        blocked_reason=blocked_reason,
        planned_operations=tuple(operations),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"preview_only": True, "executed": False, "operation_count": len(operations)},
    )
