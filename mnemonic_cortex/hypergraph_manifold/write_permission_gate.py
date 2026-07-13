"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: write permission gate.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Write-permission state construction for HGM-6.

This module never executes writes. It only creates explicit permission records
for downstream preview-only transaction planning.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

from .enums import TraceEventKind, ValidationSeverity
from .hgm6_result import HGM6CommitOptions, WritePermissionState
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def _stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def coerce_hgm6_options(options: Optional[HGM6CommitOptions | Mapping[str, Any]]) -> HGM6CommitOptions:
    if options is None:
        return HGM6CommitOptions()
    if isinstance(options, HGM6CommitOptions):
        return options
    return HGM6CommitOptions(**dict(options))


def _redact(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): _redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_redact(key, v) for v in value)
    return value


def trace_hgm6(component: str, validation: ValidationResult, payload: Optional[Mapping[str, Any]] = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): _redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


def build_write_permission_state(
    requested: bool = False,
    granted: bool = False,
    config=None,
    options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None,
) -> WritePermissionState:
    """Create an explicit preview-only write-permission state.

    ``granted=True`` only affects readiness scoring. It never executes a write.
    """

    opts = coerce_hgm6_options(options)
    req = bool(requested or opts.requested)
    grant = bool(granted or opts.granted)
    if grant and not req:
        req = True
    validation = ValidationResult()
    if not req:
        reason = "write permission not requested; default denied"
        validation.info("hgm6_permission.default_denied", reason, "requested")
    elif req and not grant:
        reason = "write permission requested but not granted"
        validation.warning("hgm6_permission.requested_not_granted", reason, "granted")
    else:
        reason = "write permission granted for preview-readiness only; no execution allowed in HGM-6"
        validation.warning("hgm6_permission.preview_only_grant", reason, "granted")
    trace = trace_hgm6("write_permission_gate.build_write_permission_state", validation, {"requested": req, "granted": grant, "dry_run": opts.dry_run, "preview_only": opts.preview_only})
    return WritePermissionState(
        permission_id=f"hgm6_perm_{_stable_hash(req, grant, opts.dry_run, opts.preview_only)}",
        requested=req,
        granted=grant,
        reason=reason,
        dry_run=bool(opts.dry_run),
        preview_only=bool(opts.preview_only),
        trace_id=trace.trace_id,
        metadata={"executed": False, "preview_only": True},
    )


def build_hgm6_write_permission_gate(memory_plan, hgm5_result=None, config=None, options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None):
    """High-level HGM-6 entry point.

    Builds permission state and a preview-only transaction commit plan. This
    function never writes into QDT/WM and never mutates the supplied plan.
    """

    from .hgm6_result import HGM6WritePermissionResult
    from .transaction_preview import build_transaction_commit_preview

    opts = coerce_hgm6_options(options)
    validation = ValidationResult()
    traces = []
    permission = build_write_permission_state(requested=opts.requested, granted=opts.granted, config=config, options=opts)
    preview = build_transaction_commit_preview(memory_plan, hgm5_result=hgm5_result, config=config, options=opts)
    validation.merge(preview.validation)
    traces.extend(preview.trace_records)
    trace = trace_hgm6("write_permission_gate.build_hgm6_write_permission_gate", validation, {"preview_id": preview.preview_id, "allowed": preview.allowed, "executed": False})
    traces.append(trace)
    return HGM6WritePermissionResult(
        write_permission=permission,
        transaction_preview=preview,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"preview_only": True, "executed": False, "allowed": preview.allowed},
    )
