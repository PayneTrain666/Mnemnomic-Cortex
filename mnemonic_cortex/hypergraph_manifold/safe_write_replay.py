"""Safe write replay evaluator for HGM-8.

Replay evaluation consumes HGM-7 logs/results and never performs live writes.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hgm7_result import TransactionLog, TransactionLogEntry, WriteExecutionResult, HGM7WriteExecutionResult
from .hgm8_result import HGM8RuntimeOptions, SafeWriteReplayRecord, SafeWriteReplayResult, hgm8_stable_hash, trace_hgm8
from .runtime_embedding_trainer import coerce_hgm8_options
from .validation import ValidationResult


def _extract_log_entries(log_or_result: Any) -> Tuple[TransactionLogEntry, ...]:
    if isinstance(log_or_result, TransactionLog):
        return tuple(log_or_result.entries or tuple())
    if isinstance(log_or_result, WriteExecutionResult):
        return tuple(log_or_result.transaction_log.entries or tuple())
    if isinstance(log_or_result, HGM7WriteExecutionResult):
        return tuple(log_or_result.execution_result.transaction_log.entries or tuple())
    if isinstance(log_or_result, TransactionLogEntry):
        return (log_or_result,)
    if isinstance(log_or_result, (list, tuple)):
        entries: List[TransactionLogEntry] = []
        for item in log_or_result:
            entries.extend(_extract_log_entries(item))
        return tuple(entries)
    return tuple()


def evaluate_safe_write_replay(
    log_or_result: Any,
    config=None,
    options: Optional[HGM8RuntimeOptions | Mapping[str, Any]] = None,
) -> SafeWriteReplayResult:
    """Replay transaction logs as safety evaluations, never as writes."""

    opts = coerce_hgm8_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    entries = _extract_log_entries(log_or_result)
    if not entries:
        validation.warning("hgm8_replay.empty", "no transaction log entries available for replay evaluation", "log_or_result")
    if len(entries) > opts.max_log_entries:
        validation.warning("hgm8_replay.bounded", "log entry count exceeded max_log_entries; truncated", "entries")
    records: List[SafeWriteReplayRecord] = []
    for entry in sorted(entries[: opts.max_log_entries], key=lambda e: (e.target_slot_id, e.operation_id, e.entry_id)):
        reason = "safe preview replay"
        safe = True
        score = 1.0
        if opts.require_dry_run and not entry.dry_run:
            safe = False
            score = 0.0
            reason = "entry is not dry-run"
            validation.error("hgm8_replay.not_dry_run", reason, entry.entry_id)
        elif entry.status == "test_executed" and not opts.allow_test_logs:
            safe = False
            score = 0.0
            reason = "test-executed log entries are not allowed by options"
            validation.error("hgm8_replay.test_log_blocked", reason, entry.entry_id)
        elif entry.status == "blocked":
            safe = True
            score = 0.75
            reason = "blocked entry is safe but not useful for replay"
        elif entry.status == "test_executed":
            safe = True
            score = 0.65
            reason = "isolated test-execution entry is replayed as log-only evidence"
        elif entry.status == "simulated":
            safe = True
            score = 1.0
            reason = "simulated entry is safe for replay evaluation"
        else:
            safe = False
            score = 0.0
            reason = f"unknown transaction status {entry.status!r}"
            validation.error("hgm8_replay.unknown_status", reason, entry.entry_id)
        trace = trace_hgm8("safe_write_replay.record", validation, {"entry_id": entry.entry_id, "status": entry.status, "safe": safe, "secret_token": "must_redact"})
        traces.append(trace)
        records.append(SafeWriteReplayRecord(
            replay_id=f"hgm8_replay_{hgm8_stable_hash(entry.entry_id, entry.status, safe)}",
            source_entry_id=entry.entry_id,
            operation_id=entry.operation_id,
            status=entry.status,
            safe=bool(safe),
            replayed=False,
            score=score,
            reason=reason,
            trace_id=trace.trace_id,
            metadata={"live_qdt_write": False, "replay_only": True, "target_slot_id": entry.target_slot_id},
        ))
    aggregate = sum(record.score for record in records) / len(records) if records else 0.0
    all_safe = bool(records) and all(record.safe for record in records)
    final_trace = trace_hgm8("safe_write_replay.evaluate_safe_write_replay", validation, {"record_count": len(records), "aggregate_score": aggregate, "safe": all_safe})
    traces.append(final_trace)
    return SafeWriteReplayResult(
        records=tuple(records),
        safe=all_safe,
        aggregate_score=aggregate,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"record_count": len(records), "live_qdt_write": False, "replay_only": True},
    )
