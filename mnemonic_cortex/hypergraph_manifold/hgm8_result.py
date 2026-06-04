"""HGM-8 result dataclasses for runtime embedding/replay/benchmark evaluation.

HGM-8 is evaluation-first. It builds deterministic runtime embedding records,
replays HGM-7 transaction logs as safe evaluation objects, and scores the
end-to-end HGM pipeline without performing live QDT/WM writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import hashlib

from .types import TraceRecord
from .validation import ValidationResult
from .enums import TraceEventKind, ValidationSeverity

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def hgm8_stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def hgm8_redact(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): hgm8_redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(hgm8_redact(key, v) for v in value)
    return value


def trace_hgm8(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): hgm8_redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


@dataclass(frozen=True)
class HGM8RuntimeOptions:
    """Options for HGM-8 deterministic runtime evaluation.

    Defaults are bounded, deterministic, and dependency-light. ``allow_test_logs``
    permits scoring explicit HGM-7 isolated test-execution log entries, but does
    not permit or perform any write execution.
    """

    max_records: int = 256
    embedding_dimension: int = 16
    deterministic_seed: int = 2028
    benchmark_iterations: int = 1
    max_log_entries: int = 256
    allow_test_logs: bool = True
    require_dry_run: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_records) <= 0:
            raise ValueError("max_records must be positive")
        if int(self.embedding_dimension) <= 0:
            raise ValueError("embedding_dimension must be positive")
        if int(self.benchmark_iterations) <= 0:
            raise ValueError("benchmark_iterations must be positive")
        if int(self.max_log_entries) <= 0:
            raise ValueError("max_log_entries must be positive")
        object.__setattr__(self, "max_records", int(self.max_records))
        object.__setattr__(self, "embedding_dimension", int(self.embedding_dimension))
        object.__setattr__(self, "deterministic_seed", int(self.deterministic_seed))
        object.__setattr__(self, "benchmark_iterations", int(self.benchmark_iterations))
        object.__setattr__(self, "max_log_entries", int(self.max_log_entries))


@dataclass(frozen=True)
class RuntimeEmbeddingRecord:
    embedding_id: str
    source_id: str
    source_type: str
    vector: Tuple[float, ...]
    confidence: float
    learned_runtime_ready: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(v) for v in (self.vector or tuple())))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))


@dataclass(frozen=True)
class RuntimeEmbeddingTrainerResult:
    embeddings: Tuple[RuntimeEmbeddingRecord, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embeddings", tuple(self.embeddings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SafeWriteReplayRecord:
    replay_id: str
    source_entry_id: str
    operation_id: str
    status: str
    safe: bool
    replayed: bool
    score: float
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))


@dataclass(frozen=True)
class SafeWriteReplayResult:
    records: Tuple[SafeWriteReplayRecord, ...]
    safe: bool
    aggregate_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or tuple()))
        object.__setattr__(self, "aggregate_score", max(0.0, min(1.0, float(self.aggregate_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class PipelineBenchmarkMetric:
    metric_id: str
    metric_name: str
    score: float
    weight: float
    explanation: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))


@dataclass(frozen=True)
class PipelineBenchmarkResult:
    metrics: Tuple[PipelineBenchmarkMetric, ...]
    aggregate_score: float
    passed: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics or tuple()))
        object.__setattr__(self, "aggregate_score", max(0.0, min(1.0, float(self.aggregate_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM8PipelineEvaluationResult:
    trainer_result: RuntimeEmbeddingTrainerResult
    replay_result: SafeWriteReplayResult
    benchmark_result: PipelineBenchmarkResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
