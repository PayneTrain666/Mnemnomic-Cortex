"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm2 result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-2 result dataclasses for manifold routing and depth retrieval.

HGM-2 routes HGM-1 bound scenario hyperedges into geometry-aware
manifold chart assignments and then creates depth-layer retrieval targets.
The layer is dependency-light and deliberately returns typed result objects
for recoverable validation failures instead of throwing runtime exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import DepthLayer, GeometryType
from .hgm1_result import BoundScenarioHyperedge
from .types import ManifoldChart, TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class ManifoldRoutingOptions:
    """Options controlling HGM-2 deterministic routing behavior."""

    allow_missing_coordinates: bool = True
    allow_unsupported_fallback: bool = False
    fallback_geometry: GeometryType = GeometryType.EUCLIDEAN
    default_depth: DepthLayer = DepthLayer.D3_RELATION
    max_coordinate_length: int = 4096
    coordinate_metadata_key: str = "coordinates"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback_geometry", GeometryType.coerce(self.fallback_geometry))
        object.__setattr__(self, "default_depth", DepthLayer.coerce(self.default_depth))
        if int(self.max_coordinate_length) <= 0:
            raise ValueError("max_coordinate_length must be positive")


@dataclass(frozen=True)
class ManifoldRoutingInput:
    """Input envelope for manifold chart routing."""

    bound_hyperedges: Tuple[BoundScenarioHyperedge, ...]
    available_manifold_charts: Tuple[ManifoldChart, ...]
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    geometry_preferences: Mapping[str, Any] = field(default_factory=dict)
    depth_hints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound_hyperedges", tuple(self.bound_hyperedges or tuple()))
        object.__setattr__(self, "available_manifold_charts", tuple(self.available_manifold_charts or tuple()))


@dataclass(frozen=True)
class GeometryDistanceResult:
    geometry_type: GeometryType
    distance: float
    similarity: float
    method: str
    validation: ValidationResult
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_type", GeometryType.coerce(self.geometry_type))


@dataclass(frozen=True)
class ManifoldRouteAssignment:
    assignment_id: str
    hyperedge_id: str
    chart_id: str
    geometry_type: GeometryType
    depth_layer: DepthLayer
    distance_score: float
    similarity_score: float
    confidence: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_type", GeometryType.coerce(self.geometry_type))
        object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))


@dataclass(frozen=True)
class ManifoldRoutingResult:
    assignments: Tuple[ManifoldRouteAssignment, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class DepthRetrievalTarget:
    target_id: str
    hyperedge_id: str
    depth_layer: DepthLayer
    retrieval_key: str
    priority: float
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))


@dataclass(frozen=True)
class DepthRetrievalBridgeResult:
    targets: Tuple[DepthRetrievalTarget, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM2ManifoldRoutingResult:
    routing: ManifoldRoutingResult
    depth_bridge: DepthRetrievalBridgeResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
