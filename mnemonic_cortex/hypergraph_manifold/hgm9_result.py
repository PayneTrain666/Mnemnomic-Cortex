"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm9 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-9 result dataclasses for runtime integration readiness.

HGM-9 is evaluation-first. It scores HGM/QDT runtime adapter readiness,
slot-lattice replay quality, and production-readiness gates without enabling
production execution or writing into QDT/WM memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import hashlib

from .types import TraceRecord
from .validation import ValidationResult
from .enums import TraceEventKind, ValidationSeverity

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def hgm9_stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def hgm9_redact(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): hgm9_redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(hgm9_redact(key, v) for v in value)
    return value


def trace_hgm9(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): hgm9_redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


@dataclass(frozen=True)
class HGM9ReadinessOptions:
    """Options for HGM-9 readiness evaluation.

    Defaults are conservative, bounded, deterministic, and evaluation-only.
    ``production_enable_allowed`` intentionally defaults to False because
    HGM-9 is a gate, not a production executor.
    """

    max_records: int = 256
    max_replay_hooks: int = 256
    readiness_threshold: float = 0.80
    require_adapter_available: bool = False
    require_pipeline_passed: bool = False
    require_no_live_write: bool = True
    production_enable_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_records) <= 0:
            raise ValueError("max_records must be positive")
        if int(self.max_replay_hooks) <= 0:
            raise ValueError("max_replay_hooks must be positive")
        object.__setattr__(self, "max_records", int(self.max_records))
        object.__setattr__(self, "max_replay_hooks", int(self.max_replay_hooks))
        object.__setattr__(self, "readiness_threshold", max(0.0, min(1.0, float(self.readiness_threshold))))


@dataclass(frozen=True)
class QDTRuntimeReadinessMetric:
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
class QDTRuntimeEvaluationResult:
    metrics: Tuple[QDTRuntimeReadinessMetric, ...]
    adapter_available: bool
    integration_score: float
    runtime_ready: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics or tuple()))
        object.__setattr__(self, "integration_score", max(0.0, min(1.0, float(self.integration_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SlotLatticeReplayRecord:
    replay_id: str
    hook_id: str
    source_record_id: str
    target_slot_id: str
    depth_layer: str
    geometry_type: str
    replayable: bool
    score: float
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))


@dataclass(frozen=True)
class SlotLatticeReplayBenchmarkResult:
    records: Tuple[SlotLatticeReplayRecord, ...]
    aggregate_score: float
    replay_safe: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or tuple()))
        object.__setattr__(self, "aggregate_score", max(0.0, min(1.0, float(self.aggregate_score))))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ProductionReadinessGate:
    gate_id: str
    score: float
    ready: bool
    production_enabled: bool
    blockers: Tuple[str, ...]
    warnings: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "blockers", tuple(self.blockers or tuple()))
        object.__setattr__(self, "warnings", tuple(self.warnings or tuple()))


@dataclass(frozen=True)
class HGM9RuntimeIntegrationResult:
    qdt_runtime_evaluation: QDTRuntimeEvaluationResult
    slot_lattice_replay_benchmark: SlotLatticeReplayBenchmarkResult
    production_readiness_gate: ProductionReadinessGate
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
