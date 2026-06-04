"""Runtime embedding trainer scaffold for HGM-8.

This module is intentionally dependency-light and deterministic. It prepares
runtime-ready embedding records from HGM-5/HGM-7 records without training a
production model or mutating QDT/WM state.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple
import math

from .hgm5_result import HGMEmbeddingRecord
from .hgm7_result import TransactionLog, TransactionLogEntry, RecoveryVerificationResult, WriteExecutionResult, HGM7WriteExecutionResult
from .hgm8_result import HGM8RuntimeOptions, RuntimeEmbeddingRecord, RuntimeEmbeddingTrainerResult, hgm8_stable_hash, trace_hgm8
from .validation import ValidationResult


def coerce_hgm8_options(options: Optional[HGM8RuntimeOptions | Mapping[str, Any]]) -> HGM8RuntimeOptions:
    if options is None:
        return HGM8RuntimeOptions()
    if isinstance(options, HGM8RuntimeOptions):
        return options
    return HGM8RuntimeOptions(**dict(options))


def _stable_float(seed: str, index: int) -> float:
    raw = int(hgm8_stable_hash(seed, index, length=12), 16)
    return (raw % 2000000) / 1000000.0 - 1.0


def _normalize(vec: Sequence[float]) -> Tuple[float, ...]:
    vals = tuple(float(v) for v in vec)
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 1e-12:
        return tuple(0.0 for _ in vals)
    return tuple(v / norm for v in vals)


def _fit_vector(vec: Sequence[float], dim: int, seed: str) -> Tuple[float, ...]:
    vals = [float(v) for v in vec if math.isfinite(float(v))]
    if len(vals) >= dim:
        return _normalize(vals[:dim])
    while len(vals) < dim:
        vals.append(_stable_float(seed, len(vals)))
    return _normalize(vals)


def _source_identity(record: Any, index: int) -> Tuple[str, str]:
    if isinstance(record, HGMEmbeddingRecord):
        return record.embedding_id or record.source_id or f"record_{index}", "HGMEmbeddingRecord"
    if isinstance(record, TransactionLogEntry):
        return record.entry_id or record.operation_id or f"record_{index}", "TransactionLogEntry"
    if isinstance(record, TransactionLog):
        return record.log_id or f"record_{index}", "TransactionLog"
    if isinstance(record, RecoveryVerificationResult):
        return f"recovery_{len(record.records)}_{index}", "RecoveryVerificationResult"
    if isinstance(record, WriteExecutionResult):
        return record.execution_id or f"record_{index}", "WriteExecutionResult"
    if isinstance(record, HGM7WriteExecutionResult):
        return record.execution_result.execution_id or f"record_{index}", "HGM7WriteExecutionResult"
    return f"unsupported_{index}", type(record).__name__


def _confidence(record: Any) -> float:
    if isinstance(record, HGMEmbeddingRecord):
        return max(0.0, min(1.0, float(record.confidence)))
    if isinstance(record, TransactionLogEntry):
        if record.status == "simulated":
            return 0.85
        if record.status == "test_executed":
            return 0.65
        return 0.35
    if isinstance(record, TransactionLog):
        return 0.85 if record.complete else 0.35
    if isinstance(record, RecoveryVerificationResult):
        return 0.9 if record.rollback_ready else 0.3
    if isinstance(record, WriteExecutionResult):
        return 0.8 if record.allowed and not record.executed else 0.55 if record.executed else 0.3
    if isinstance(record, HGM7WriteExecutionResult):
        return _confidence(record.execution_result)
    return 0.0


def _vector_from_record(record: Any, opts: HGM8RuntimeOptions, source_id: str, source_type: str) -> Tuple[float, ...]:
    if isinstance(record, HGMEmbeddingRecord):
        base = tuple(float(v) for v in record.vector)
        return _fit_vector(base, opts.embedding_dimension, f"hgm8|{source_id}|{opts.deterministic_seed}")
    if isinstance(record, TransactionLogEntry):
        status_value = {"blocked": -0.5, "simulated": 0.5, "test_executed": 0.25}.get(record.status, 0.0)
        flags = [status_value, 1.0 if record.dry_run else -1.0, 1.0 if record.simulation_mode else -1.0, float(len(record.operation_type or "")) / 32.0]
        return _fit_vector(flags, opts.embedding_dimension, f"hgm8|{source_id}|{opts.deterministic_seed}")
    if isinstance(record, TransactionLog):
        entries = tuple(record.entries or tuple())
        simulated = sum(1 for e in entries if e.status == "simulated")
        blocked = sum(1 for e in entries if e.status == "blocked")
        test = sum(1 for e in entries if e.status == "test_executed")
        base = [len(entries) / max(1, opts.max_log_entries), simulated / max(1, len(entries)), blocked / max(1, len(entries)), test / max(1, len(entries)), 1.0 if record.complete else -1.0]
        return _fit_vector(base, opts.embedding_dimension, f"hgm8|{source_id}|{opts.deterministic_seed}")
    if isinstance(record, RecoveryVerificationResult):
        records = tuple(record.records or tuple())
        verified = sum(1 for r in records if r.verified)
        base = [len(records) / max(1, opts.max_log_entries), verified / max(1, len(records)), 1.0 if record.rollback_ready else -1.0]
        return _fit_vector(base, opts.embedding_dimension, f"hgm8|{source_id}|{opts.deterministic_seed}")
    if isinstance(record, WriteExecutionResult):
        base = [1.0 if record.allowed else -1.0, 1.0 if record.simulation_mode else -1.0, 1.0 if record.executed else -1.0, len(record.transaction_log.entries) / max(1, opts.max_log_entries), 1.0 if record.recovery_verification.rollback_ready else -1.0]
        return _fit_vector(base, opts.embedding_dimension, f"hgm8|{source_id}|{opts.deterministic_seed}")
    if isinstance(record, HGM7WriteExecutionResult):
        return _vector_from_record(record.execution_result, opts, source_id, source_type)
    return tuple()


def build_runtime_embedding_trainer(
    records: Sequence[Any],
    config=None,
    options: Optional[HGM8RuntimeOptions | Mapping[str, Any]] = None,
) -> RuntimeEmbeddingTrainerResult:
    """Build deterministic runtime embedding records from HGM-5/HGM-7 records."""

    opts = coerce_hgm8_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    recs = tuple(records or tuple())
    if not recs:
        validation.warning("hgm8_embeddings.empty", "no records supplied; no runtime embeddings generated", "records")
    if len(recs) > opts.max_records:
        validation.warning("hgm8_embeddings.bounded", "record count exceeded max_records; truncated", "records")
    supported = (HGMEmbeddingRecord, TransactionLog, TransactionLogEntry, RecoveryVerificationResult, WriteExecutionResult, HGM7WriteExecutionResult)
    embeddings: List[RuntimeEmbeddingRecord] = []
    for idx, record in enumerate(recs[: opts.max_records]):
        if not isinstance(record, supported):
            validation.warning("hgm8_embeddings.unsupported_record", f"unsupported record skipped: {type(record).__name__}", f"records[{idx}]")
            continue
        source_id, source_type = _source_identity(record, idx)
        vector = _vector_from_record(record, opts, source_id, source_type)
        if len(vector) != opts.embedding_dimension or not all(math.isfinite(v) for v in vector):
            validation.error("hgm8_embeddings.invalid_vector", "runtime embedding vector is non-finite or wrong length", f"records[{idx}]")
            continue
        trace = trace_hgm8("runtime_embedding_trainer.build_runtime_embedding_trainer", validation, {"source_id": source_id, "source_type": source_type, "metadata": getattr(record, "metadata", {})})
        traces.append(trace)
        embeddings.append(RuntimeEmbeddingRecord(
            embedding_id=f"hgm8_rt_emb_{hgm8_stable_hash(source_type, source_id, opts.embedding_dimension, opts.deterministic_seed)}",
            source_id=str(source_id),
            source_type=source_type,
            vector=tuple(vector),
            confidence=_confidence(record),
            learned_runtime_ready=False,
            trace_id=trace.trace_id,
            metadata={"trainer_mode": "deterministic_runtime_scaffold", "source_index": idx, "evaluation_first": True},
        ))
    embeddings.sort(key=lambda emb: (emb.source_type, emb.source_id, emb.embedding_id))
    final_trace = trace_hgm8("runtime_embedding_trainer.build_runtime_embedding_trainer", validation, {"embedding_count": len(embeddings), "secret_token": "must_redact"})
    traces.append(final_trace)
    return RuntimeEmbeddingTrainerResult(
        embeddings=tuple(embeddings),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"embedding_count": len(embeddings), "max_records": opts.max_records, "embedding_dimension": opts.embedding_dimension, "evaluation_first": True},
    )
