"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: types.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Foundation dataclasses for Hypergraph Manifold / HPME.

These are deliberately small, strict, and serializable. Runtime model code
can later wrap them with Torch tensors, kernels, adapters, and planners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
import time
import uuid

from .config import HGMConfig
from .enums import (
    DepthLayer,
    GeometryType,
    HyperedgeKind,
    MutationDirection,
    ProbabilityNormalizationMode,
    TraceEventKind,
    ValidationSeverity,
)
from .shapes import TensorShapeContract, finite_number, iter_leaf_values, nested_shape
from .validation import ValidationResult


def _stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _coerce_depth_sequence(values: Optional[Sequence[Any]]) -> Tuple[DepthLayer, ...]:
    if values is None:
        return tuple()
    return tuple(DepthLayer.coerce(v) for v in values)


@dataclass(frozen=True)
class MagnitudeBin:
    bin_id: str
    lower: float
    upper: float
    label: str = ""

    @property
    def center(self) -> float:
        return (float(self.lower) + float(self.upper)) / 2.0

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if not self.bin_id:
            result.error("magnitude_bin.missing_id", "MagnitudeBin.bin_id is required", "bin_id")
        if not finite_number(self.lower):
            result.error("magnitude_bin.invalid_lower", "lower must be finite", "lower")
        if not finite_number(self.upper):
            result.error("magnitude_bin.invalid_upper", "upper must be finite", "upper")
        if finite_number(self.lower) and finite_number(self.upper) and float(self.lower) > float(self.upper):
            result.error("magnitude_bin.invalid_range", "lower must be <= upper", "lower")
        return result


@dataclass(frozen=True)
class ManifoldChart:
    chart_id: str
    geometry: GeometryType
    dimension: int
    coordinate_system: str = "local"
    curvature: Optional[float] = None
    metric_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", GeometryType.coerce(self.geometry))

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.chart_id:
            result.error("chart.missing_id", "ManifoldChart.chart_id is required", "chart_id")
        if self.geometry not in config.allowed_geometries:
            result.error("chart.geometry_not_allowed", f"Geometry {self.geometry.value!r} is not allowed", "geometry")
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            result.error("chart.invalid_dimension", "dimension must be a positive integer", "dimension")
        if self.curvature is not None and not finite_number(self.curvature):
            result.error("chart.invalid_curvature", "curvature must be finite when provided", "curvature")
        return result


@dataclass(frozen=True)
class DepthLayerAssignment:
    target_id: str
    layer: DepthLayer
    geometry: Optional[GeometryType] = None
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", DepthLayer.coerce(self.layer))
        if self.geometry is not None:
            object.__setattr__(self, "geometry", GeometryType.coerce(self.geometry))

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.target_id:
            result.error("depth_assignment.missing_target", "target_id is required", "target_id")
        if not (finite_number(self.confidence) and 0.0 <= float(self.confidence) <= 1.0):
            result.error("depth_assignment.invalid_confidence", "confidence must be finite and in [0, 1]", "confidence")
        if self.geometry is not None and self.geometry not in config.allowed_geometries:
            result.error("depth_assignment.geometry_not_allowed", f"Geometry {self.geometry.value!r} is not allowed", "geometry")
        return result


@dataclass(frozen=True)
class QSpinSignature:
    signature_id: str
    components: Tuple[float, ...]
    phase: float = 0.0
    basis: str = "canonical"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.signature_id:
            result.error("qspin.missing_id", "signature_id is required", "signature_id")
        if len(self.components) != config.qspin_dimension:
            result.error(
                "qspin.dimension_mismatch",
                f"expected {config.qspin_dimension} components, got {len(self.components)}",
                "components",
            )
        for idx, value in enumerate(self.components):
            if not finite_number(value):
                result.error("qspin.non_finite_component", f"component {idx} is not finite", f"components[{idx}]")
        if not finite_number(self.phase):
            result.error("qspin.non_finite_phase", "phase must be finite", "phase")
        return result


@dataclass(frozen=True)
class MutationToken:
    token_id: str
    variable_id: str
    direction: MutationDirection
    magnitude_bin_id: str
    probability: Optional[float] = None
    depth_layer: Optional[DepthLayer] = None
    geometry: Optional[GeometryType] = None
    context_id: Optional[str] = None
    action_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", MutationDirection.coerce(self.direction))
        if self.depth_layer is not None:
            object.__setattr__(self, "depth_layer", DepthLayer.coerce(self.depth_layer))
        if self.geometry is not None:
            object.__setattr__(self, "geometry", GeometryType.coerce(self.geometry))

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.token_id:
            result.error("mutation_token.missing_id", "token_id is required", "token_id")
        if not self.variable_id:
            result.error("mutation_token.missing_variable", "variable_id is required", "variable_id")
        if not self.magnitude_bin_id:
            result.error("mutation_token.missing_magnitude_bin", "magnitude_bin_id is required", "magnitude_bin_id")
        if self.probability is not None:
            if not (finite_number(self.probability) and 0.0 <= float(self.probability) <= 1.0):
                result.error("mutation_token.invalid_probability", "probability must be finite and in [0, 1]", "probability")
        if self.geometry is not None and self.geometry not in config.allowed_geometries:
            result.error("mutation_token.geometry_not_allowed", f"Geometry {self.geometry.value!r} is not allowed", "geometry")
        return result


@dataclass(frozen=True)
class ScenarioHyperedge:
    hyperedge_id: str
    node_ids: Tuple[str, ...]
    kind: HyperedgeKind = HyperedgeKind.SCENARIO
    weight: float = 1.0
    mutation_token_ids: Tuple[str, ...] = tuple()
    geometry: Optional[GeometryType] = None
    depth_layers: Tuple[DepthLayer, ...] = tuple()
    qspin_signature_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "kind", HyperedgeKind.coerce(self.kind))
        object.__setattr__(self, "mutation_token_ids", tuple(self.mutation_token_ids or tuple()))
        object.__setattr__(self, "depth_layers", _coerce_depth_sequence(self.depth_layers))
        if self.geometry is not None:
            object.__setattr__(self, "geometry", GeometryType.coerce(self.geometry))

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.hyperedge_id:
            result.error("hyperedge.missing_id", "hyperedge_id is required", "hyperedge_id")
        min_nodes = 1 if config.allow_singleton_hyperedges else 2
        if len(self.node_ids) < min_nodes:
            result.error("hyperedge.too_few_nodes", f"hyperedge requires at least {min_nodes} node(s)", "node_ids")
        if len(self.node_ids) > config.max_hyperedge_nodes:
            result.error("hyperedge.too_many_nodes", "hyperedge exceeds max_hyperedge_nodes", "node_ids")
        if len(set(self.node_ids)) != len(self.node_ids):
            result.error("hyperedge.duplicate_nodes", "node_ids must be unique", "node_ids")
        if not (finite_number(self.weight) and 0.0 <= float(self.weight) <= 1.0):
            result.error("hyperedge.invalid_weight", "weight must be finite and in [0, 1]", "weight")
        if self.geometry is not None and self.geometry not in config.allowed_geometries:
            result.error("hyperedge.geometry_not_allowed", f"Geometry {self.geometry.value!r} is not allowed", "geometry")
        return result


@dataclass(frozen=True)
class HypersetMatrix:
    matrix_id: str
    probabilities: Any
    contract: TensorShapeContract
    variable_ids: Tuple[str, ...]
    magnitude_bins: Tuple[MagnitudeBin, ...]
    normalization_mode: ProbabilityNormalizationMode = ProbabilityNormalizationMode.MUTATION_AXIS
    depth_layers: Tuple[DepthLayer, ...] = tuple()
    context_ids: Tuple[str, ...] = tuple()
    time_ids: Tuple[str, ...] = tuple()
    action_ids: Tuple[str, ...] = tuple()
    tolerance: float = 1e-5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_ids", tuple(self.variable_ids))
        object.__setattr__(self, "magnitude_bins", tuple(self.magnitude_bins))
        object.__setattr__(self, "normalization_mode", ProbabilityNormalizationMode.coerce(self.normalization_mode))
        object.__setattr__(self, "depth_layers", _coerce_depth_sequence(self.depth_layers))
        object.__setattr__(self, "context_ids", tuple(self.context_ids or tuple()))
        object.__setattr__(self, "time_ids", tuple(self.time_ids or tuple()))
        object.__setattr__(self, "action_ids", tuple(self.action_ids or tuple()))

    @property
    def shape(self) -> Tuple[int, ...]:
        return nested_shape(self.probabilities)

    def validate(self, config: HGMConfig = HGMConfig()) -> ValidationResult:
        result = ValidationResult()
        if not self.matrix_id:
            result.error("hyperset_matrix.missing_id", "matrix_id is required", "matrix_id")
        if not (finite_number(self.tolerance) and float(self.tolerance) > 0):
            result.error("hyperset_matrix.invalid_tolerance", "tolerance must be positive and finite", "tolerance")

        # Shape checks.
        try:
            shape = self.shape
            result.merge(self.contract.validate_shape(shape))
        except ValueError as exc:
            result.error("hyperset_matrix.ragged_probabilities", str(exc), "probabilities")
            return result

        axis_lengths = {
            "v": len(self.variable_ids),
            "m": len(self.magnitude_bins),
            "d": len(self.depth_layers),
            "c": len(self.context_ids),
            "t": len(self.time_ids),
            "a": len(self.action_ids),
        }
        for idx, axis in enumerate(self.contract.axes):
            expected = axis_lengths.get(axis)
            actual = shape[idx] if idx < len(shape) else None
            if expected is not None and expected > 0 and actual != expected:
                result.error(
                    "hyperset_matrix.axis_metadata_mismatch",
                    f"Axis {axis!r} expects metadata length {expected}, got tensor size {actual}",
                    f"axis.{axis}",
                )
            if expected == 0 and axis in {"d", "c", "t", "a"}:
                result.error(
                    "hyperset_matrix.missing_axis_metadata",
                    f"Axis {axis!r} is present in {self.contract.name} but no metadata IDs were provided",
                    f"axis.{axis}",
                )

        # Validate bins.
        seen_bins = set()
        for index, bin_item in enumerate(self.magnitude_bins):
            result.merge(bin_item.validate())
            if bin_item.bin_id in seen_bins:
                result.error("hyperset_matrix.duplicate_magnitude_bin", f"duplicate bin_id {bin_item.bin_id!r}", f"magnitude_bins[{index}]")
            seen_bins.add(bin_item.bin_id)

        # Value checks: only nested Python data can be fully inspected.
        if isinstance(self.probabilities, (list, tuple)):
            for coords, value in iter_leaf_values(self.probabilities):
                path = "probabilities" + "".join(f"[{i}]" for i in coords)
                if not finite_number(value):
                    result.error("probability.non_finite", "probability value must be finite", path)
                elif float(value) < 0.0:
                    result.error("probability.negative", "probability value must be non-negative", path)
            if result.ok:
                self._validate_normalization(result)
        else:
            result.warning("probability.values_not_inspected", "probability object exposes shape but values were not inspected", "probabilities")

        return result

    def _validate_normalization(self, result: ValidationResult) -> None:
        tol = float(self.tolerance)
        values = [(coords, float(value)) for coords, value in iter_leaf_values(self.probabilities)]
        if self.normalization_mode == ProbabilityNormalizationMode.NONE:
            return
        if self.normalization_mode == ProbabilityNormalizationMode.GLOBAL:
            total = sum(value for _, value in values)
            if abs(total - 1.0) > tol:
                result.error("probability.global_not_normalized", f"global probability sum is {total}, expected 1.0", "probabilities")
            return

        # ROW and MUTATION_AXIS both normalize across mutation axis for HGM-0A.
        mutation_axis = self.contract.mutation_axis_index
        groups: Dict[Tuple[int, ...], float] = {}
        for coords, value in values:
            group_key = tuple(c for idx, c in enumerate(coords) if idx != mutation_axis)
            groups[group_key] = groups.get(group_key, 0.0) + value
        for group_key, total in groups.items():
            if abs(total - 1.0) > tol:
                result.error(
                    "probability.mutation_axis_not_normalized",
                    f"mutation-axis group {group_key} sums to {total}, expected 1.0",
                    "probabilities",
                )


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    event_kind: TraceEventKind
    component: str
    timestamp_utc: float = field(default_factory=lambda: time.time())
    severity: ValidationSeverity = ValidationSeverity.INFO
    lineage: Tuple[str, ...] = ("MnemonicCortex", "HGM", "HPME", "HGM-0A")
    parent_trace_id: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_kind", TraceEventKind.coerce(self.event_kind))
        object.__setattr__(self, "severity", ValidationSeverity.coerce(self.severity))
        object.__setattr__(self, "lineage", tuple(self.lineage))

    @classmethod
    def create(
        cls,
        event_kind: TraceEventKind,
        component: str,
        *,
        severity: ValidationSeverity = ValidationSeverity.INFO,
        parent_trace_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> "TraceRecord":
        return cls(
            trace_id=_stable_id("trace"),
            event_kind=event_kind,
            component=component,
            severity=severity,
            parent_trace_id=parent_trace_id,
            payload=dict(payload or {}),
        )

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if not self.trace_id:
            result.error("trace.missing_id", "trace_id is required", "trace_id")
        if not self.component:
            result.error("trace.missing_component", "component is required", "component")
        if not finite_number(self.timestamp_utc) or float(self.timestamp_utc) <= 0:
            result.error("trace.invalid_timestamp", "timestamp_utc must be positive and finite", "timestamp_utc")
        if not self.lineage:
            result.error("trace.missing_lineage", "lineage cannot be empty", "lineage")
        return result

    def redacted_payload(self) -> Dict[str, Any]:
        """Return payload with obvious secret-bearing keys removed.

        This supports the reliability/security layer without pretending to be
        a complete DLP system.
        """

        blocked = {"secret", "token", "api_key", "password", "credential"}
        return {
            key: ("<redacted>" if any(term in key.lower() for term in blocked) else value)
            for key, value in dict(self.payload).items()
        }
