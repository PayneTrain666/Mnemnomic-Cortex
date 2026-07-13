"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: transaction log.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Transaction-log generation for HGM-7 guarded write execution.

The log records simulated/blocked/test-isolated operation attempts. It never
performs QDT/WM writes by itself.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from .hgm6_result import TransactionCommitPreview
from .hgm7_result import (
    HGM7ExecutionOptions,
    TransactionLog,
    TransactionLogEntry,
    hgm7_stable_hash,
    trace_hgm7,
)
from .validation import ValidationResult


def coerce_hgm7_options(options: Optional[HGM7ExecutionOptions | Mapping[str, Any]]) -> HGM7ExecutionOptions:
    if options is None:
        return HGM7ExecutionOptions()
    if isinstance(options, HGM7ExecutionOptions):
        return options
    return HGM7ExecutionOptions(**dict(options))


def build_transaction_log(
    commit_preview: TransactionCommitPreview,
    config=None,
    options: Optional[HGM7ExecutionOptions | Mapping[str, Any]] = None,
) -> TransactionLog:
    """Build deterministic transaction log entries from an HGM-6 preview.

    Status values:
    - ``simulated``: structurally allowed and running in simulation mode.
    - ``test_executed``: isolated test execution path explicitly enabled.
    - ``blocked``: operation or preview is not allowed.
    """

    opts = coerce_hgm7_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    if not isinstance(commit_preview, TransactionCommitPreview):
        validation.error("hgm7_log.invalid_preview", "commit_preview must be a TransactionCommitPreview", "commit_preview")
        trace = trace_hgm7("transaction_log.build_transaction_log", validation, {"reason": "invalid_preview"})
        return TransactionLog("hgm7_log_invalid", tuple(), False, validation, (trace,), metadata={"executed": False})

    operations = tuple(commit_preview.operations or tuple())
    if len(operations) > opts.max_operations:
        validation.warning("hgm7_log.bounded_operation_count", "operation count exceeded max_operations; log entries truncated", "operations")
    operations = tuple(sorted(operations[: opts.max_operations], key=lambda op: (op.target_slot_id, op.operation_id)))
    if not operations:
        validation.warning("hgm7_log.empty_operations", "no operations available for transaction log", "operations")

    preview_blocked = opts.require_commit_preview_allowed and not commit_preview.allowed
    entries: List[TransactionLogEntry] = []
    for op in operations:
        blocked_reason = ""
        structurally_allowed = bool(op.allowed and not preview_blocked)
        if preview_blocked:
            blocked_reason = commit_preview.blocked_reason or "commit preview is not allowed"
        elif not op.allowed:
            blocked_reason = op.blocked_reason or "operation preview is blocked"

        if not structurally_allowed:
            status = "blocked"
            executed_flag = False
            validation.warning("hgm7_log.operation_blocked", blocked_reason, op.operation_id)
        elif opts.simulation_mode:
            status = "simulated"
            executed_flag = False
        elif opts.allow_test_execution:
            status = "test_executed"
            executed_flag = True
            validation.warning("hgm7_log.test_execution", "isolated test execution enabled; no QDT/WM live write is performed by default adapter", op.operation_id)
        else:
            status = "blocked"
            executed_flag = False
            blocked_reason = "non-simulation execution requested without allow_test_execution"
            validation.error("hgm7_log.execution_not_allowed", blocked_reason, op.operation_id)

        trace = trace_hgm7("transaction_log.entry", validation, {
            "operation_id": op.operation_id,
            "status": status,
            "blocked_reason": blocked_reason,
            "target_slot_id": op.target_slot_id,
            "secret_token": "must_redact",
        })
        traces.append(trace)
        entries.append(TransactionLogEntry(
            entry_id=f"hgm7_log_entry_{hgm7_stable_hash(op.operation_id, status, op.target_slot_id)}",
            operation_id=op.operation_id,
            operation_type=op.operation_type,
            status=status,
            source_payload_id=op.source_payload_id,
            target_slot_id=op.target_slot_id,
            dry_run=True,
            simulation_mode=bool(opts.simulation_mode),
            trace_id=trace.trace_id,
            metadata={"executed": executed_flag, "blocked_reason": blocked_reason, "preview_only": True},
        ))

    complete = bool(entries) and all(entry.status in {"simulated", "test_executed", "blocked"} for entry in entries)
    final_trace = trace_hgm7("transaction_log.build_transaction_log", validation, {"entry_count": len(entries), "complete": complete})
    traces.append(final_trace)
    return TransactionLog(
        log_id=f"hgm7_log_{hgm7_stable_hash(getattr(commit_preview, 'preview_id', 'invalid'), tuple(e.entry_id for e in entries))}",
        entries=tuple(entries),
        complete=complete,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"entry_count": len(entries), "simulation_mode": opts.simulation_mode, "executed": any(e.metadata.get("executed") for e in entries)},
    )
