"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: geometry distance.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Dependency-light geometry distance wrappers for HGM-2.

The wrappers intentionally favor validation and deterministic behavior over
high-performance kernels. Learned embeddings, Torch adapters, and optimized
manifold kernels are later-stage work.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from .config import HGMConfig
from .enums import GeometryType, TraceEventKind, ValidationSeverity
from .hgm2_result import GeometryDistanceResult
from .shapes import finite_number
from .types import TraceRecord
from .validation import ValidationResult

_EPS = 1e-12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _coerce_point(point: Any, *, path: str, result: ValidationResult, max_len: int = 4096) -> Tuple[float, ...]:
    if point is None:
        result.error("hgm2_geometry.missing_point", f"{path} is required", path)
        return tuple()
    if hasattr(point, "tolist"):
        point = point.tolist()
    if isinstance(point, (int, float)):
        point = [point]
    if not isinstance(point, (list, tuple)):
        result.error("hgm2_geometry.invalid_point", f"{path} must be a sequence of finite numbers", path)
        return tuple()
    if len(point) == 0:
        result.error("hgm2_geometry.empty_point", f"{path} cannot be empty", path)
        return tuple()
    if len(point) > max_len:
        result.error("hgm2_geometry.point_too_long", f"{path} exceeds max coordinate length {max_len}", path)
        return tuple()
    values: List[float] = []
    for idx, value in enumerate(point):
        if not finite_number(value):
            result.error("hgm2_geometry.non_finite_coordinate", "coordinate values must be finite", f"{path}[{idx}]")
            continue
        values.append(float(value))
    return tuple(values)


def _same_length(a: Sequence[float], b: Sequence[float], result: ValidationResult) -> bool:
    if len(a) != len(b):
        result.error("hgm2_geometry.dimension_mismatch", f"point dimensions differ: {len(a)} != {len(b)}", "point")
        return False
    return True


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _trace(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    severity = ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=severity,
        payload=dict(payload or {}),
    )


def _finish(geometry: GeometryType, distance: float, similarity: float, method: str, validation: ValidationResult, metadata=None) -> GeometryDistanceResult:
    distance = float(distance) if finite_number(distance) else float("inf")
    similarity = _clamp(float(similarity) if finite_number(similarity) else 0.0, 0.0, 1.0)
    if validation.ok and (distance < 0 or not finite_number(distance)):
        validation.error("hgm2_geometry.invalid_distance", "distance must be finite and non-negative", "distance")
    if validation.ok and not finite_number(similarity):
        validation.error("hgm2_geometry.invalid_similarity", "similarity must be finite", "similarity")
    trace = _trace("geometry_distance.compute_geometry_distance", validation, {"geometry": geometry.value, "method": method})
    return GeometryDistanceResult(geometry, distance, similarity, method, validation, trace.trace_id, dict(metadata or {}))


def _euclidean(a: Sequence[float], b: Sequence[float], validation: ValidationResult):
    d = _l2(a, b)
    return d, 1.0 / (1.0 + d), "euclidean_l2"


def _spherical(a: Sequence[float], b: Sequence[float], validation: ValidationResult):
    na, nb = _norm(a), _norm(b)
    if na <= _EPS or nb <= _EPS:
        validation.error("hgm2_geometry.zero_norm", "spherical distance requires non-zero vectors", "point")
        return float("inf"), 0.0, "spherical_angular"
    cosine = _clamp(_dot(a, b) / max(_EPS, na * nb), -1.0, 1.0)
    angular = math.acos(cosine)
    similarity = (cosine + 1.0) / 2.0
    return angular, similarity, "spherical_cosine_angular"


def _hyperbolic(a: Sequence[float], b: Sequence[float], validation: ValidationResult):
    na2 = sum(x * x for x in a)
    nb2 = sum(x * x for x in b)
    if na2 >= 1.0 or nb2 >= 1.0:
        validation.error("hgm2_geometry.outside_poincare_ball", "hyperbolic points must be inside the unit Poincare ball", "point")
        return float("inf"), 0.0, "poincare_ball_safe"
    delta2 = sum((x - y) ** 2 for x, y in zip(a, b))
    denom = max(_EPS, (1.0 - na2) * (1.0 - nb2))
    arg = max(1.0, 1.0 + 2.0 * delta2 / denom)
    d = math.acosh(arg)
    return d, 1.0 / (1.0 + d), "poincare_ball_safe"


def _torus(a: Sequence[float], b: Sequence[float], validation: ValidationResult):
    deltas = []
    for x, y in zip(a, b):
        raw = abs((x % 1.0) - (y % 1.0))
        deltas.append(min(raw, 1.0 - raw))
    d = math.sqrt(sum(delta * delta for delta in deltas))
    return d, 1.0 / (1.0 + d), "torus_wrapped_l2"


def _as_complex(values: Sequence[float]) -> Tuple[complex, ...]:
    if len(values) >= 2 and len(values) % 2 == 0:
        return tuple(complex(values[i], values[i + 1]) for i in range(0, len(values), 2))
    return tuple(complex(v, 0.0) for v in values)


def _complex_projective(a: Sequence[float], b: Sequence[float], validation: ValidationResult):
    ca = _as_complex(a)
    cb = _as_complex(b)
    if len(ca) != len(cb):
        validation.error("hgm2_geometry.dimension_mismatch", "complex projective vectors differ after complex pairing", "point")
        return float("inf"), 0.0, "complex_projective_phase_invariant"
    na = math.sqrt(sum(abs(x) ** 2 for x in ca))
    nb = math.sqrt(sum(abs(x) ** 2 for x in cb))
    if na <= _EPS or nb <= _EPS:
        validation.error("hgm2_geometry.zero_norm", "complex projective distance requires non-zero vectors", "point")
        return float("inf"), 0.0, "complex_projective_phase_invariant"
    inner = sum(x.conjugate() * y for x, y in zip(ca, cb))
    similarity = _clamp(abs(inner) / max(_EPS, na * nb), 0.0, 1.0)
    return 1.0 - similarity, similarity, "complex_projective_phase_invariant"


def _product(point_a: Any, point_b: Any, validation: ValidationResult, config: HGMConfig):
    # Preferred rich shape: [{"geometry": "euclidean", "a": [...], "b": [...], "weight": 1.0}, ...]
    # Fallback simple shape: treat the supplied vectors as a Euclidean product component and warn.
    components = None
    if isinstance(point_a, Mapping) and "components" in point_a:
        components = point_a.get("components")
    elif isinstance(point_b, Mapping) and "components" in point_b:
        components = point_b.get("components")
    if components:
        weighted_d = 0.0
        weighted_s = 0.0
        total_w = 0.0
        for idx, component in enumerate(components):
            if not isinstance(component, Mapping):
                validation.warning("hgm2_geometry.product_component_skipped", "product component must be a mapping", f"components[{idx}]")
                continue
            geom = component.get("geometry", GeometryType.EUCLIDEAN)
            weight = component.get("weight", 1.0)
            if not finite_number(weight) or float(weight) <= 0:
                validation.warning("hgm2_geometry.product_component_bad_weight", "component weight must be positive", f"components[{idx}].weight")
                continue
            sub = compute_geometry_distance(component.get("a"), component.get("b"), geom, config=config)
            validation.merge(sub.validation)
            if sub.validation.ok:
                weighted_d += float(weight) * sub.distance
                weighted_s += float(weight) * sub.similarity
                total_w += float(weight)
        if total_w <= 0:
            validation.error("hgm2_geometry.product_no_valid_components", "product geometry has no valid components", "components")
            return float("inf"), 0.0, "product_weighted_components"
        return weighted_d / total_w, weighted_s / total_w, "product_weighted_components"
    validation.warning("hgm2_geometry.product_fallback", "product component metadata missing; using Euclidean fallback", "point")
    a = _coerce_point(point_a, path="point_a", result=validation)
    b = _coerce_point(point_b, path="point_b", result=validation)
    if validation.ok and _same_length(a, b, validation):
        d, s, _ = _euclidean(a, b, validation)
        return d, s, "product_euclidean_fallback"
    return float("inf"), 0.0, "product_euclidean_fallback"


def compute_geometry_distance(point_a, point_b, geometry_type, config: HGMConfig = HGMConfig()) -> GeometryDistanceResult:
    """Compute a deterministic dependency-light distance for a supported geometry."""

    validation = ValidationResult()
    try:
        geometry = GeometryType.coerce(geometry_type)
    except ValueError:
        validation.error("hgm2_geometry.unsupported_geometry", f"unsupported geometry {geometry_type!r}", "geometry_type")
        trace = _trace("geometry_distance.compute_geometry_distance", validation, {"geometry": repr(geometry_type)})
        return GeometryDistanceResult(GeometryType.EUCLIDEAN, float("inf"), 0.0, "unsupported_geometry", validation, trace.trace_id)

    if geometry not in config.allowed_geometries:
        validation.error("hgm2_geometry.geometry_not_allowed", f"geometry {geometry.value!r} is not allowed", "geometry_type")
        return _finish(geometry, float("inf"), 0.0, "geometry_not_allowed", validation)

    if geometry == GeometryType.PRODUCT:
        d, s, method = _product(point_a, point_b, validation, config)
        return _finish(geometry, d, s, method, validation)

    a = _coerce_point(point_a, path="point_a", result=validation)
    b = _coerce_point(point_b, path="point_b", result=validation)
    if not validation.ok or not _same_length(a, b, validation):
        return _finish(geometry, float("inf"), 0.0, "invalid_points", validation)

    if geometry == GeometryType.EUCLIDEAN:
        d, s, method = _euclidean(a, b, validation)
    elif geometry == GeometryType.SPHERICAL:
        d, s, method = _spherical(a, b, validation)
    elif geometry == GeometryType.HYPERBOLIC:
        d, s, method = _hyperbolic(a, b, validation)
    elif geometry == GeometryType.TORUS:
        d, s, method = _torus(a, b, validation)
    elif geometry == GeometryType.COMPLEX_PROJECTIVE:
        d, s, method = _complex_projective(a, b, validation)
    else:
        validation.error("hgm2_geometry.unsupported_geometry", f"unsupported geometry {geometry.value!r}", "geometry_type")
        d, s, method = float("inf"), 0.0, "unsupported_geometry"
    return _finish(geometry, d, s, method, validation)
