"""HGM-5 result dataclasses for embedding/evaluation scaffolding.

HGM-5 is evaluation-first: it builds deterministic dependency-light
embeddings and quality scores for HGM bridge records without mutating QDT/WM
internals or performing live memory writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class EmbeddingTrainerOptions:
    """Options for dependency-light deterministic embedding generation."""

    max_records: int = 256
    embedding_dimension: int = 16
    deterministic_seed: int = 1337
    allow_optional_numpy: bool = False
    allow_optional_torch: bool = False
    train_mode: str = "deterministic_baseline"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_records) <= 0:
            raise ValueError("max_records must be positive")
        if int(self.embedding_dimension) <= 0:
            raise ValueError("embedding_dimension must be positive")
        object.__setattr__(self, "max_records", int(self.max_records))
        object.__setattr__(self, "embedding_dimension", int(self.embedding_dimension))
        object.__setattr__(self, "deterministic_seed", int(self.deterministic_seed))
        object.__setattr__(self, "train_mode", str(self.train_mode or "deterministic_baseline"))


@dataclass(frozen=True)
class HGMEmbeddingRecord:
    embedding_id: str
    source_id: str
    source_type: str
    vector: Tuple[float, ...]
    confidence: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(x) for x in (self.vector or tuple())))


@dataclass(frozen=True)
class EmbeddingTrainerResult:
    embeddings: Tuple[HGMEmbeddingRecord, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embeddings", tuple(self.embeddings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class BridgeQualityMetric:
    metric_id: str
    source_id: str
    metric_name: str
    score: float
    weight: float
    explanation: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeEvaluationResult:
    metrics: Tuple[BridgeQualityMetric, ...]
    aggregate_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class IntegrationScore:
    score_id: str
    source_id: str
    source_type: str
    score: float
    confidence: float
    explanation: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationScoringResult:
    scores: Tuple[IntegrationScore, ...]
    aggregate_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", tuple(self.scores or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM5EmbeddingEvaluationResult:
    trainer_result: EmbeddingTrainerResult
    bridge_evaluation_result: BridgeEvaluationResult
    integration_scoring_result: IntegrationScoringResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
