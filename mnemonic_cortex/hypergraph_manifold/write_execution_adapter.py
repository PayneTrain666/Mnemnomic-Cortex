"""Guarded write-execution adapter scaffold for HGM-7.

Default behavior is simulation-mode only. No QDT/WM memory is mutated here.
Optional test execution records are isolated and log-only unless a caller wraps
this module with a separate, explicit adapter in a later stage.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .hgm6_result import TransactionCommitPreview
from .hgm7_result import (
    HGM7ExecutionOptions,
    HGM7WriteExecutionResult,
    WriteExecutionAdapterStatus,
    WriteExecutionResult,
    hgm7_stable_hash,
    trace_hgm7,
)
from .recovery_verification import verify_recovery
from .transaction_log import build_transaction_log, coerce_hgm7_options
from .validation import ValidationResult


def build_write_execution_adapter_status(
    config=None,
    options: Optional[HGM7ExecutionOptions | Mapping[str, Any]] = None,
) -> WriteExecutionAdapterStatus:
    """Return explicit adapter status without importing or mutating QDT/WM."""

    opts = coerce_hgm7_options(options)
    validation = ValidationResult()
    if opts.simulation_mode:
        available = True
        reason = "simulation-mode adapter available; live writes disabled"
        validation.info("hgm7_adapter.simulation_available", reason, "simulation_mode")
    elif opts.allow_test_execution:
        available = True
        reason = "isolated test execution adapter enabled; no default QDT/WM write path"
        validation.warning("hgm7_adapter.test_execution_enabled", reason, "allow_test_execution")
    else:
        available = False
        reason = "non-simulation execution requested without allow_test_execution"
        validation.error("hgm7_adapter.unavailable", reason, "allow_test_execution")
    trace = trace_hgm7("write_execution_adapter.build_write_execution_adapter_status", validation, {
        "available": available,
        "simulation_mode": opts.simulation_mode,
        "allow_test_execution": opts.allow_test_execution,
    })
    return WriteExecutionAdapterStatus(
        adapter_id=f"hgm7_adapter_{hgm7_stable_hash(opts.test_adapter_id, opts.simulation_mode, opts.allow_test_execution)}",
        available=available,
        simulation_mode=bool(opts.simulation_mode),
        reason=reason,
        trace_id=trace.trace_id,
        metadata={"test_adapter_id": opts.test_adapter_id, "live_qdt_write": False},
    )


def execute_write_adapter(
    commit_preview: TransactionCommitPreview,
    config=None,
    options: Optional[HGM7ExecutionOptions | Mapping[str, Any]] = None,
) -> WriteExecutionResult:
    """Build log and recovery verification for a guarded write adapter run.

    This function never performs a live QDT/WM write. ``executed`` is true only
    for explicit isolated test execution logs; simulation mode returns
    ``executed=False``.
    """

    opts = coerce_hgm7_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    adapter_status = build_write_execution_adapter_status(config=config, options=opts)
    if not isinstance(commit_preview, TransactionCommitPreview):
        validation.error("hgm7_execute.invalid_preview", "commit_preview must be a TransactionCommitPreview", "commit_preview")
        log = build_transaction_log(commit_preview, config=config, options=opts)
        recovery = verify_recovery(getattr(commit_preview, "rollback_manifest", None), log, config=config, options=opts)
        validation.merge(log.validation).merge(recovery.validation)
        traces.extend(log.trace_records + recovery.trace_records)
        trace = trace_hgm7("write_execution_adapter.execute_write_adapter", validation, {"allowed": False, "reason": "invalid_preview"})
        traces.append(trace)
        return WriteExecutionResult(
            execution_id=f"hgm7_exec_{hgm7_stable_hash('invalid', trace.trace_id)}",
            adapter_status=adapter_status,
            transaction_log=log,
            recovery_verification=recovery,
            executed=False,
            simulation_mode=opts.simulation_mode,
            allowed=False,
            blocked_reason="invalid commit preview",
            validation=validation,
            trace_records=tuple(traces),
            metadata={"preview_only": True, "live_qdt_write": False},
        )

    log = build_transaction_log(commit_preview, config=config, options=opts)
    recovery = verify_recovery(commit_preview.rollback_manifest, log, config=config, options=opts)
    validation.merge(log.validation).merge(recovery.validation)
    traces.extend(log.trace_records + recovery.trace_records)

    executed = any(entry.status == "test_executed" for entry in log.entries)
    blocked_reason = ""
    allowed = bool(adapter_status.available and log.complete and recovery.rollback_ready and (opts.simulation_mode or opts.allow_test_execution))
    if not adapter_status.available:
        allowed = False
        blocked_reason = adapter_status.reason
    elif opts.require_commit_preview_allowed and not commit_preview.allowed:
        allowed = False
        blocked_reason = commit_preview.blocked_reason or "commit preview was not allowed"
    elif opts.require_rollback_verified and not recovery.rollback_ready:
        allowed = False
        blocked_reason = "rollback verification failed"
    elif not opts.simulation_mode and not opts.allow_test_execution:
        allowed = False
        blocked_reason = "non-simulation execution not enabled"

    if blocked_reason:
        validation.warning("hgm7_execute.blocked", blocked_reason, "execution")
    trace = trace_hgm7("write_execution_adapter.execute_write_adapter", validation, {
        "allowed": allowed,
        "executed": executed,
        "simulation_mode": opts.simulation_mode,
        "blocked_reason": blocked_reason,
        "live_qdt_write": False,
    })
    traces.append(trace)
    return WriteExecutionResult(
        execution_id=f"hgm7_exec_{hgm7_stable_hash(commit_preview.preview_id, log.log_id, recovery.rollback_ready, executed)}",
        adapter_status=adapter_status,
        transaction_log=log,
        recovery_verification=recovery,
        executed=executed,
        simulation_mode=bool(opts.simulation_mode),
        allowed=allowed,
        blocked_reason=blocked_reason,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"preview_only": True, "live_qdt_write": False, "test_execution": executed},
    )


def build_hgm7_write_execution_adapter(
    commit_preview: TransactionCommitPreview,
    config=None,
    options: Optional[HGM7ExecutionOptions | Mapping[str, Any]] = None,
) -> HGM7WriteExecutionResult:
    """High-level HGM-7 entry point."""

    validation = ValidationResult()
    traces: List[Any] = []
    execution = execute_write_adapter(commit_preview, config=config, options=options)
    validation.merge(execution.validation)
    traces.extend(execution.trace_records)
    trace = trace_hgm7("write_execution_adapter.build_hgm7_write_execution_adapter", validation, {
        "execution_id": execution.execution_id,
        "allowed": execution.allowed,
        "executed": execution.executed,
        "live_qdt_write": False,
    })
    traces.append(trace)
    return HGM7WriteExecutionResult(
        execution_result=execution,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"preview_only": True, "live_qdt_write": False, "executed": execution.executed},
    )
