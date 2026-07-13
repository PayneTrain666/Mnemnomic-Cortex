"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt wm bridge.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
QDT/working-memory adapter detection and high-level HGM-4 entry point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .enums import TraceEventKind, ValidationSeverity
from .hgm4_result import BridgeAdapterStatus, HGM4BridgeResult, QDTWMBridgeOptions
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
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.WARNING,
        payload=dict(payload or {}),
    )


def detect_qdt_wm_adapter_status(config=None, options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None) -> BridgeAdapterStatus:
    """Detect whether expected QDT/WM paths are present without heavy imports.

    Uses ``importlib.util.find_spec`` and filesystem checks only. It never
    imports QDT/WM runtime modules and therefore cannot trigger model loading.
    """

    opts = _coerce_options(options)
    validation = ValidationResult()
    detected = []
    # Module specs only; no imports.
    for mod in opts.expected_module_paths:
        try:
            if importlib.util.find_spec(mod) is not None:
                detected.append(f"module:{mod}")
        except Exception:
            # Broken packages should not crash bridge planning.
            continue
    # Filesystem checks relative to cwd and its parents where practical.
    cwd = Path.cwd()
    roots = [cwd, *cwd.parents]
    seen = set()
    for rel in opts.expected_filesystem_paths:
        for root in roots[:4]:
            path = (root / rel).resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                detected.append(f"path:{path}")
                break
    available = bool(detected)
    if not available:
        validation.warning("hgm4_adapter.unavailable", "QDT/WM adapter paths were not detected", "adapter_status")
    reason = "detected expected QDT/WM package path(s)" if available else "expected QDT/WM package paths unavailable; dry-run plan can still be produced"
    trace = _trace("qdt_wm_bridge.detect_qdt_wm_adapter_status", validation, {"available": available, "detected_count": len(detected)})
    return BridgeAdapterStatus(
        adapter_id=f"hgm4_adapter_{_stable_hash(tuple(detected), available)}",
        available=available,
        adapter_type="qdt_wm_dry_run_detector",
        reason=reason,
        detected_paths=tuple(sorted(detected)),
        trace_id=trace.trace_id,
        metadata={"dry_run_only": True, "heavy_imports": False},
    )


def build_hgm4_qdt_wm_bridge(records: Sequence[Any], config=None, options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None) -> HGM4BridgeResult:
    """High-level HGM-4 entry point for trace-safe bridge planning."""

    # Import lazily to avoid a circular dependency with the detector function.
    from .trace_safe_memory_integration import build_trace_safe_memory_plan, preview_bridge_execution

    plan = build_trace_safe_memory_plan(records, config=config, options=options)
    preview = preview_bridge_execution(plan, config=config, options=options)
    validation = ValidationResult()
    validation.merge(plan.validation).merge(preview.validation)
    traces = tuple(plan.trace_records) + tuple(preview.trace_records)
    return HGM4BridgeResult(
        adapter_status=plan.adapter_status,
        memory_plan=plan,
        execution_preview=preview,
        validation=validation,
        trace_records=traces,
        metadata={"dry_run": plan.dry_run, "write_intent": plan.write_intent, "preview_allowed": preview.allowed},
    )
