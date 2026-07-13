"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm1 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-1 result dataclasses for scenario hyperedge binding.

This layer converts HGM-0B ``ScenarioCandidate`` atoms into bounded,
traceable hyperedge structures. It remains dependency-light and avoids
Torch/Numpy requirements so it can be used in dry-run validation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import HyperedgeKind
from .runtime_result import ScenarioCandidate
from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class HyperedgeBindingOptions:
    """Options controlling HGM-1 deterministic candidate grouping."""

    group_by: Tuple[str, ...] = ("context_id", "time_index", "action_id", "depth")
    max_hyperedge_size: Optional[int] = None
    hyperedge_kind: HyperedgeKind = HyperedgeKind.SCENARIO
    include_singletons: bool = False
    relation_hint_key: str = "relation_hints"

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_by", tuple(self.group_by or tuple()))
        object.__setattr__(self, "hyperedge_kind", HyperedgeKind.coerce(self.hyperedge_kind))


@dataclass(frozen=True)
class HyperedgeBindingInput:
    """Input envelope for hyperedge binding."""

    candidates: Tuple[ScenarioCandidate, ...]
    source_payload_metadata: Mapping[str, Any] = field(default_factory=dict)
    variable_relation_hints: Mapping[Tuple[str, str], float] = field(default_factory=dict)
    context_hints: Mapping[str, Any] = field(default_factory=dict)
    action_hints: Mapping[str, Any] = field(default_factory=dict)
    time_hints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates or tuple()))


@dataclass(frozen=True)
class BoundScenarioHyperedge:
    """A typed, scored HGM-1 scenario hyperedge."""

    hyperedge_id: str
    candidate_ids: Tuple[str, ...]
    node_ids: Tuple[str, ...]
    kind: HyperedgeKind
    coherence_score: float
    probability_score: float
    conflict_score: float
    opportunity_score: float
    source_candidate_indices: Tuple[Tuple[int, ...], ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids or tuple()))
        object.__setattr__(self, "node_ids", tuple(self.node_ids or tuple()))
        object.__setattr__(self, "kind", HyperedgeKind.coerce(self.kind))
        object.__setattr__(self, "source_candidate_indices", tuple(tuple(int(x) for x in coords) for coords in self.source_candidate_indices))


@dataclass(frozen=True)
class CoherenceScoreReport:
    score: float
    method: str
    evidence_count: int
    warnings: Tuple[str, ...] = tuple()
    errors: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings or tuple()))
        object.__setattr__(self, "errors", tuple(self.errors or tuple()))


@dataclass(frozen=True)
class ConflictEdge:
    conflict_id: str
    left_candidate_id: str
    right_candidate_id: str
    variable_id: str
    reason: str
    score: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpportunityEdge:
    opportunity_id: str
    candidate_ids: Tuple[str, ...]
    reason: str
    score: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids or tuple()))


@dataclass(frozen=True)
class HyperedgeBindingResult:
    hyperedges: Tuple[BoundScenarioHyperedge, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperedges", tuple(self.hyperedges or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ConflictGraphResult:
    conflict_edges: Tuple[ConflictEdge, ...]
    conflict_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflict_edges", tuple(self.conflict_edges or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class OpportunityGraphResult:
    opportunity_edges: Tuple[OpportunityEdge, ...]
    opportunity_score: float
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "opportunity_edges", tuple(self.opportunity_edges or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM1ScenarioGraphResult:
    binding: HyperedgeBindingResult
    conflict_graph: ConflictGraphResult
    opportunity_graph: OpportunityGraphResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
