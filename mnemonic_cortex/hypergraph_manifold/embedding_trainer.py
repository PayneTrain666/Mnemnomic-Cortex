"""Dependency-light HGM embedding trainer scaffold.

This is not a production learned trainer. It builds deterministic baseline
embeddings from HGM bridge records for evaluation, scoring, and future
learned-runtime preparation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, is_dataclass
from typing import Any, List, Mapping, Optional, Sequence

from .enums import TraceEventKind, ValidationSeverity
from .hgm4_result import BridgeExecutionPreview, HGMBridgePayload, SharedSlotLatticeHook, TraceSafeMemoryPlan
from .hgm5_result import EmbeddingTrainerOptions, EmbeddingTrainerResult, HGMEmbeddingRecord
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def _stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _coerce_options(options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]]) -> EmbeddingTrainerOptions:
    if options is None:
        return EmbeddingTrainerOptions()
    if isinstance(options, EmbeddingTrainerOptions):
        return options
    return EmbeddingTrainerOptions(**dict(options))


def _redact_value(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_redact_value(key, v) for v in value)
    return value


def _safe_mapping(record: Any) -> Mapping[str, Any]:
    try:
        if is_dataclass(record):
            data = asdict(record)
        elif isinstance(record, Mapping):
            data = dict(record)
        else:
            data = {"repr": repr(record)}
    except Exception:
        data = {"repr": repr(record)}
    return {str(k): _redact_value(str(k), v) for k, v in data.items()}


def _trace(component: str, validation: ValidationResult, payload: Optional[Mapping[str, Any]] = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): _redact_value(str(k), v) for k, v in dict(payload or {}).items()},
    )


def _source_identity(record: Any, idx: int = 0) -> tuple[str, str]:
    if isinstance(record, HGMBridgePayload):
        return record.source_id, "HGMBridgePayload"
    if isinstance(record, SharedSlotLatticeHook):
        return record.source_record_id, "SharedSlotLatticeHook"
    if isinstance(record, TraceSafeMemoryPlan):
        return record.plan_id, "TraceSafeMemoryPlan"
    if isinstance(record, BridgeExecutionPreview):
        return record.preview_id, "BridgeExecutionPreview"
    return getattr(record, "source_id", None) or getattr(record, "payload_id", None) or f"unsupported_{idx}", type(record).__name__


def _vector_from_record(record: Any, opts: EmbeddingTrainerOptions) -> tuple[float, ...]:
    data = _safe_mapping(record)
    # Deterministic signed random projection from stable textual payload.
    base = repr(sorted(data.items(), key=lambda kv: kv[0]))
    values: List[float] = []
    for dim in range(opts.embedding_dimension):
        digest = hashlib.sha256(f"{opts.deterministic_seed}|{dim}|{base}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "big", signed=False)
        unit = (integer / float(2**64 - 1)) * 2.0 - 1.0
        values.append(unit)
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0 or not math.isfinite(norm):
        return tuple(0.0 for _ in range(opts.embedding_dimension))
    return tuple(float(v / norm) for v in values)


def _confidence_for(record: Any) -> float:
    if isinstance(record, HGMBridgePayload):
        return 0.75 if record.qspin_signature_id else 0.55
    if isinstance(record, SharedSlotLatticeHook):
        return max(0.0, min(1.0, float(record.confidence)))
    if isinstance(record, TraceSafeMemoryPlan):
        return 0.8 if record.dry_run and not record.write_intent else 0.4
    if isinstance(record, BridgeExecutionPreview):
        return 0.9 if (not record.metadata.get("executed", False)) else 0.1
    return 0.0


def _optional_dependency_metadata(opts: EmbeddingTrainerOptions) -> Mapping[str, Any]:
    found = {}
    if opts.allow_optional_numpy:
        try:
            __import__("numpy")
            found["numpy_available"] = True
        except Exception:
            found["numpy_available"] = False
    else:
        found["numpy_available"] = "not_requested"
    if opts.allow_optional_torch:
        try:
            __import__("torch")
            found["torch_available"] = True
        except Exception:
            found["torch_available"] = False
    else:
        found["torch_available"] = "not_requested"
    return found


def build_baseline_hgm_embeddings(records: Sequence[Any], config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> EmbeddingTrainerResult:
    """Build deterministic baseline embeddings from HGM bridge records."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    recs = tuple(records or tuple())
    if not recs:
        validation.warning("hgm5_embeddings.empty", "no records supplied; no embeddings generated", "records")
    if len(recs) > opts.max_records:
        validation.warning("hgm5_embeddings.bounded", "record count exceeded max_records; truncated", "records")
    dep_meta = _optional_dependency_metadata(opts)
    embeddings: List[HGMEmbeddingRecord] = []
    supported = (HGMBridgePayload, SharedSlotLatticeHook, TraceSafeMemoryPlan, BridgeExecutionPreview)
    for idx, record in enumerate(recs[: opts.max_records]):
        if not isinstance(record, supported):
            validation.warning("hgm5_embeddings.unsupported_record", f"unsupported record type skipped: {type(record).__name__}", f"records[{idx}]")
            continue
        source_id, source_type = _source_identity(record, idx)
        vector = _vector_from_record(record, opts)
        if len(vector) != opts.embedding_dimension or not all(math.isfinite(v) for v in vector):
            validation.error("hgm5_embeddings.invalid_vector", "embedding vector is non-finite or wrong length", f"records[{idx}]")
            continue
        trace = _trace("embedding_trainer.build_baseline_hgm_embeddings", validation, {"source_id": source_id, "source_type": source_type, "metadata": getattr(record, "metadata", {})})
        traces.append(trace)
        embeddings.append(HGMEmbeddingRecord(
            embedding_id=f"hgm5_emb_{_stable_hash(source_type, source_id, opts.embedding_dimension, opts.deterministic_seed)}",
            source_id=str(source_id),
            source_type=source_type,
            vector=vector,
            confidence=_confidence_for(record),
            trace_id=trace.trace_id,
            metadata={"train_mode": opts.train_mode, "dependency_mode": "pure_python", "source_index": idx},
        ))
    embeddings.sort(key=lambda e: (e.source_type, e.source_id, e.embedding_id))
    final_trace = _trace("embedding_trainer.build_baseline_hgm_embeddings", validation, {"embedding_count": len(embeddings), **dep_meta})
    traces.append(final_trace)
    return EmbeddingTrainerResult(
        embeddings=tuple(embeddings),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"embedding_count": len(embeddings), "max_records": opts.max_records, **dep_meta, "evaluation_first": True},
    )
