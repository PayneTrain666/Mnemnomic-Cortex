"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: slot lattice hooks.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Shared slot-lattice hook contracts for HGM-4.

Hooks are dry-run bridge contracts: they identify where an HGM payload would
land in a shared slot lattice, but do not mutate any memory store.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional, Sequence

from .enums import TraceEventKind, ValidationSeverity
from .hgm4_result import HGMBridgePayload, QDTWMBridgeOptions, SharedSlotLatticeHook, SharedSlotLatticeHookResult
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


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _target_slot_id(payload: HGMBridgePayload) -> str:
    return f"hgm_slot_{payload.depth_layer.value}_{_stable_hash(payload.source_type, payload.source_id, payload.geometry_type.value, payload.qspin_signature_id, length=12)}"


def build_shared_slot_lattice_hooks(
    payloads: Sequence[HGMBridgePayload],
    config=None,
    options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None,
) -> SharedSlotLatticeHookResult:
    """Build deterministic dry-run shared slot-lattice hook contracts."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    payload_tuple = tuple(payloads or tuple())
    if not payload_tuple:
        validation.warning("hgm4_hooks.empty_payloads", "empty bridge payload list; no hooks generated", "payloads")
        trace = _trace("slot_lattice_hooks.build_shared_slot_lattice_hooks", validation, {"reason": "empty_payloads"})
        return SharedSlotLatticeHookResult(tuple(), validation, (trace,), metadata={"hook_count": 0})
    if len(payload_tuple) > opts.max_hook_count:
        validation.warning("hgm4_hooks.bounded_hook_count", "payload count exceeded max_hook_count; hooks were truncated", "payloads")

    hooks: List[SharedSlotLatticeHook] = []
    for idx, payload in enumerate(payload_tuple[: opts.max_hook_count]):
        if not isinstance(payload, HGMBridgePayload):
            validation.error("hgm4_hooks.invalid_payload", "payloads must contain HGMBridgePayload records", f"payloads[{idx}]")
            continue
        if not payload.payload_id or not payload.source_id:
            validation.error("hgm4_hooks.invalid_payload_id", "payload and source IDs must be stable and non-empty", f"payloads[{idx}]")
            continue
        target_slot = _target_slot_id(payload)
        hook_id = f"hgm4_hook_{_stable_hash(payload.payload_id, target_slot)}"
        confidence = _clamp01(0.65 + 0.05 * min(payload.depth_layer.value, 7))
        trace = _trace(
            "slot_lattice_hooks.build_shared_slot_lattice_hooks",
            validation,
            {"payload_id": payload.payload_id, "target_slot_id": target_slot, "dry_run": opts.dry_run, "write_intent": opts.write_intent},
        )
        traces.append(trace)
        hooks.append(SharedSlotLatticeHook(
            hook_id=hook_id,
            source_record_id=payload.source_id,
            target_slot_id=target_slot,
            depth_layer=payload.depth_layer,
            geometry_type=payload.geometry_type,
            qspin_signature_id=payload.qspin_signature_id,
            dry_run=bool(opts.dry_run),
            write_intent=bool(opts.write_intent),
            confidence=confidence,
            trace_id=trace.trace_id,
            metadata={
                "payload_id": payload.payload_id,
                "source_type": payload.source_type,
                "bridge_contract_only": True,
            },
        ))

    hooks.sort(key=lambda h: (h.depth_layer.value, h.target_slot_id, h.source_record_id, h.hook_id))
    final_trace = _trace("slot_lattice_hooks.build_shared_slot_lattice_hooks", validation, {"hook_count": len(hooks)})
    traces.append(final_trace)
    return SharedSlotLatticeHookResult(tuple(hooks), validation, tuple(traces), metadata={"hook_count": len(hooks), "max_hook_count": opts.max_hook_count})
