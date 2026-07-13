"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm3 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-3 result dataclasses for SPCP procedural memory.

HGM-3 converts routed HGM scenario records into advisory procedural
memory records. It is dependency-light, deterministic, and intentionally
contains no live robotics-control execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class SPCPProceduralOptions:
    """Options controlling deterministic SPCP procedural-memory behavior."""

    max_sequence_length: int = 32
    max_parameter_count: int = 64
    default_duration: float = 1.0
    default_confidence: float = 0.5
    conformal_warp_bound: float = 0.1
    embedding_dimension: int = 16
    allow_zero_spherical_fallback: bool = False
    top_k: int = 5

    def __post_init__(self) -> None:
        if int(self.max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive")
        if int(self.max_parameter_count) <= 0:
            raise ValueError("max_parameter_count must be positive")
        if float(self.default_duration) < 0:
            raise ValueError("default_duration must be non-negative")
        if not (0.0 <= float(self.default_confidence) <= 1.0):
            raise ValueError("default_confidence must be in [0, 1]")
        if not (0.0 <= float(self.conformal_warp_bound) <= 0.1):
            raise ValueError("conformal_warp_bound must be in [0, 0.1]")
        if int(self.embedding_dimension) <= 0:
            raise ValueError("embedding_dimension must be positive")


@dataclass(frozen=True)
class ActionPrimitive:
    primitive_id: str
    action_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    duration: float = 1.0
    confidence: float = 1.0
    frame_id: Optional[str] = None
    object_id: Optional[str] = None
    tool_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProceduralActionSequence:
    sequence_id: str
    primitives: Tuple[ActionPrimitive, ...]
    source_hyperedge_id: str
    source_assignment_id: str
    source_depth_target_id: str
    goal_label: str
    confidence: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitives", tuple(self.primitives or tuple()))


@dataclass(frozen=True)
class ActionSequenceBuildResult:
    sequence: Optional[ProceduralActionSequence]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SPCPProcedureEmbedding:
    embedding_id: str
    sequence_id: str
    spherical_state: Tuple[float, ...]
    projective_state: Tuple[float, ...]
    conformal_warp: Tuple[float, ...]
    similarity_ready: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spherical_state", tuple(float(x) for x in (self.spherical_state or tuple())))
        object.__setattr__(self, "projective_state", tuple(float(x) for x in (self.projective_state or tuple())))
        object.__setattr__(self, "conformal_warp", tuple(float(x) for x in (self.conformal_warp or tuple())))


@dataclass(frozen=True)
class SPCPEmbeddingResult:
    embedding: Optional[SPCPProcedureEmbedding]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SPCPSimilarityResult:
    similarity: float
    distance: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ProceduralMemoryStoreResult:
    stored_sequences: Tuple[ProceduralActionSequence, ...]
    stored_embeddings: Tuple[SPCPProcedureEmbedding, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stored_sequences", tuple(self.stored_sequences or tuple()))
        object.__setattr__(self, "stored_embeddings", tuple(self.stored_embeddings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ProceduralMemoryRetrievalCandidate:
    candidate_id: str
    sequence_id: str
    similarity: float
    distance: float
    confidence: float
    source_trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProceduralMemoryRetrievalResult:
    candidates: Tuple[ProceduralMemoryRetrievalCandidate, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RoboticsPlanningActionOption:
    option_id: str
    sequence_id: str
    action_primitives: Tuple[ActionPrimitive, ...]
    expected_goal: str
    risk_score: float
    confidence: float
    explanation: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_primitives", tuple(self.action_primitives or tuple()))


@dataclass(frozen=True)
class RoboticsPlanningBridgeResult:
    action_options: Tuple[RoboticsPlanningActionOption, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_options", tuple(self.action_options or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM3ProceduralMemoryResult:
    store_result: ProceduralMemoryStoreResult
    retrieval_result: ProceduralMemoryRetrievalResult
    planning_bridge_result: RoboticsPlanningBridgeResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
