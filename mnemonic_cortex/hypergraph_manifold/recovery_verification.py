"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: recovery verification.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Recovery verification harness for HGM-7 transaction logs.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .hgm6_result import RollbackManifest
from .hgm7_result import (
    HGM7ExecutionOptions,
    RecoveryVerificationRecord,
    RecoveryVerificationResult,
    TransactionLog,
    hgm7_stable_hash,
    trace_hgm7,
)
from .transaction_log import coerce_hgm7_options
from .validation import ValidationResult


def verify_recovery(
    rollback_manifest: RollbackManifest,
    transaction_log: TransactionLog,
    config=None,
    options: Optional[HGM7ExecutionOptions | Mapping[str, Any]] = None,
) -> RecoveryVerificationResult:
    """Verify that every non-blocked transaction log entry has rollback coverage."""

    opts = coerce_hgm7_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    if not isinstance(transaction_log, TransactionLog):
        validation.error("hgm7_recovery.invalid_log", "transaction_log must be a TransactionLog", "transaction_log")
        trace = trace_hgm7("recovery_verification.verify_recovery", validation, {"reason": "invalid_log"})
        return RecoveryVerificationResult(tuple(), False, validation, (trace,), metadata={"rollback_ready": False})
    if not isinstance(rollback_manifest, RollbackManifest):
        validation.error("hgm7_recovery.invalid_manifest", "rollback_manifest must be a RollbackManifest", "rollback_manifest")
        trace = trace_hgm7("recovery_verification.verify_recovery", validation, {"reason": "invalid_manifest"})
        return RecoveryVerificationResult(tuple(), False, validation, (trace,), metadata={"rollback_ready": False})

    rb_by_op = {rb.operation_id: rb for rb in tuple(rollback_manifest.operations or tuple())}
    records: List[RecoveryVerificationRecord] = []
    for entry in sorted(transaction_log.entries, key=lambda item: (item.target_slot_id, item.operation_id)):
        needs_rollback = entry.status in {"simulated", "test_executed"}
        rb = rb_by_op.get(entry.operation_id)
        if not needs_rollback:
            verified = True
            reason = "blocked operation does not require rollback"
            rollback_id = "not_required"
        elif rb is None:
            verified = False
            reason = "missing rollback operation"
            rollback_id = "missing"
            validation.error("hgm7_recovery.missing_rollback", reason, entry.operation_id)
        elif not rb.previous_state_ref:
            verified = False
            reason = "rollback operation missing previous_state_ref"
            rollback_id = rb.rollback_id
            validation.error("hgm7_recovery.missing_previous_state", reason, rb.rollback_id)
        else:
            verified = True
            reason = "rollback coverage verified"
            rollback_id = rb.rollback_id
        trace = trace_hgm7("recovery_verification.record", validation, {"operation_id": entry.operation_id, "verified": verified, "reason": reason})
        traces.append(trace)
        records.append(RecoveryVerificationRecord(
            verification_id=f"hgm7_recovery_{hgm7_stable_hash(entry.operation_id, rollback_id, verified)}",
            rollback_id=rollback_id,
            operation_id=entry.operation_id,
            target_slot_id=entry.target_slot_id,
            verified=verified,
            reason=reason,
            trace_id=trace.trace_id,
            metadata={"needs_rollback": needs_rollback, "preview_only": True},
        ))

    relevant = tuple(record for record in records if record.rollback_id != "not_required")
    rollback_ready = bool(rollback_manifest.complete and all(record.verified for record in records))
    if opts.require_rollback_verified and relevant and not rollback_ready:
        validation.error("hgm7_recovery.not_ready", "rollback verification did not pass for all relevant operations", "rollback_manifest")
    if not records:
        validation.warning("hgm7_recovery.empty_records", "no recovery records generated", "transaction_log")
    final_trace = trace_hgm7("recovery_verification.verify_recovery", validation, {"record_count": len(records), "rollback_ready": rollback_ready})
    traces.append(final_trace)
    return RecoveryVerificationResult(
        records=tuple(records),
        rollback_ready=rollback_ready,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"record_count": len(records), "rollback_ready": rollback_ready, "preview_only": True},
    )
