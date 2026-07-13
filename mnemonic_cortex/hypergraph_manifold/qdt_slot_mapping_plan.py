"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt slot mapping plan.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Read-only HGM slot ID to QDT/WM slot mapping plan.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Optional

from .hgm4_result import SharedSlotLatticeHook, TraceSafeMemoryPlan
from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    SlotIDMappingPlan,
    SlotIDMappingRecord,
    trace_write_prep,
    write_prep_stable_hash,
)
from .qdt_write_contract_probe import _coerce_options
from .validation import ValidationResult


def _hooks_from_input(hooks_or_plan: Any) -> tuple[SharedSlotLatticeHook, ...]:
    if isinstance(hooks_or_plan, TraceSafeMemoryPlan):
        return tuple(hooks_or_plan.slot_hooks or tuple())
    if isinstance(hooks_or_plan, SharedSlotLatticeHook):
        return (hooks_or_plan,)
    if isinstance(hooks_or_plan, Iterable) and not isinstance(hooks_or_plan, (str, bytes, Mapping)):
        return tuple(h for h in hooks_or_plan if isinstance(h, SharedSlotLatticeHook))
    return tuple()


def sanitize_hgm_slot_id_for_wm(hgm_target_slot_id: str) -> str:
    raw = str(hgm_target_slot_id or "hgm_slot_missing").strip()
    raw = raw.replace("hgm_slot_", "hgm_")
    raw = re.sub(r"[^A-Za-z0-9_\-:.]", "_", raw)
    if not raw.startswith("wm_"):
        raw = f"wm_{raw}"
    return raw[:96]


def canonical_css_preview(namespace: str, local_slot_id: str, content_fingerprint: str = "") -> str:
    try:
        from mnemonic_cortex.working_memory.wm_shared_slot_registry import canonical_slot_id
        return canonical_slot_id(namespace, local_slot_id, content_fingerprint)
    except Exception:
        key = f"{namespace}|{local_slot_id}|{content_fingerprint or ''}".encode("utf-8")
        return f"css-{hashlib.sha256(key).hexdigest()[:24]}"


def build_slot_id_mapping_plan(hooks_or_plan: Any, namespace: str = "hgm_qdt_write_prep", config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> SlotIDMappingPlan:
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    hooks = _hooks_from_input(hooks_or_plan)
    if not hooks:
        validation.warning("slot_mapping.empty_hooks", "no SharedSlotLatticeHook records supplied", "slot_hooks")
    if len(hooks) > opts.max_hooks:
        validation.warning("slot_mapping.bounded_hooks", "hook count exceeded max_hooks; mappings truncated", "slot_hooks")
    mappings = []
    for hook in sorted(hooks[: opts.max_hooks], key=lambda h: (h.depth_layer.value, h.target_slot_id, h.source_record_id, h.hook_id)):
        local_slot_id = sanitize_hgm_slot_id_for_wm(hook.target_slot_id)
        content_fingerprint = write_prep_stable_hash(hook.source_record_id, hook.target_slot_id, hook.depth_layer.value, hook.geometry_type.value, length=24)
        canonical_id = canonical_css_preview(namespace, local_slot_id, content_fingerprint)
        if not canonical_id.startswith("css-"):
            validation.error("slot_mapping.invalid_canonical_id", "canonical slot preview must start with css-", hook.hook_id)
        trace = trace_write_prep("qdt_slot_mapping_plan.mapping", validation, {"hook_id": hook.hook_id, "local_slot_id": local_slot_id, "canonical_id": canonical_id})
        traces.append(trace)
        mappings.append(SlotIDMappingRecord(
            mapping_id=f"slot_map_{write_prep_stable_hash(hook.hook_id, canonical_id)}",
            source_hook_id=hook.hook_id,
            hgm_target_slot_id=hook.target_slot_id,
            wm_local_slot_id=local_slot_id,
            wm_canonical_slot_id=canonical_id,
            namespace=namespace,
            content_fingerprint=content_fingerprint,
            trace_id=trace.trace_id,
            metadata={"read_only": True, "creates_registry_record": False, "memory_type": "wm"},
        ))
    final_trace = trace_write_prep("qdt_slot_mapping_plan.build_slot_id_mapping_plan", validation, {"mapping_count": len(mappings), "namespace": namespace})
    traces.append(final_trace)
    return SlotIDMappingPlan(
        plan_id=f"slot_mapping_plan_{write_prep_stable_hash(namespace, len(mappings))}",
        mappings=tuple(mappings),
        namespace=namespace,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"read_only": True, "registry_mutated": False},
    )
