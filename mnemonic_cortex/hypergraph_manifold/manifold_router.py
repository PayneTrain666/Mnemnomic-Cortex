"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: manifold router.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-2 manifold chart router.

Routes HGM-1 ``BoundScenarioHyperedge`` records into compatible manifold
charts. The router is deterministic, dependency-light, and fail-closed for
invalid chart/hyperedge records while degrading safely for missing optional
coordinates.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import HGMConfig
from .enums import DepthLayer, GeometryType, TraceEventKind, ValidationSeverity
from .geometry_distance import compute_geometry_distance
from .hgm1_result import BoundScenarioHyperedge
from .hgm2_result import (
    ManifoldRouteAssignment,
    ManifoldRoutingInput,
    ManifoldRoutingOptions,
    ManifoldRoutingResult,
)
from .shapes import finite_number
from .types import ManifoldChart, TraceRecord
from .validation import ValidationResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_options(options: Optional[ManifoldRoutingOptions | Mapping[str, Any]]) -> ManifoldRoutingOptions:
    if options is None:
        return ManifoldRoutingOptions()
    if isinstance(options, ManifoldRoutingOptions):
        return options
    return ManifoldRoutingOptions(**dict(options))


def _coerce_input(hyperedges_or_input, charts=None, routing_options: Optional[ManifoldRoutingOptions | Mapping[str, Any]] = None):
    if isinstance(hyperedges_or_input, ManifoldRoutingInput):
        return hyperedges_or_input, _coerce_options(routing_options)
    return ManifoldRoutingInput(tuple(hyperedges_or_input or tuple()), tuple(charts or tuple())), _coerce_options(routing_options)


def _coerce_chart(raw: Any, idx: int, validation: ValidationResult) -> Optional[ManifoldChart]:
    if isinstance(raw, ManifoldChart):
        return raw
    if isinstance(raw, Mapping):
        try:
            return ManifoldChart(**dict(raw))
        except Exception as exc:  # fail closed for bad user-supplied chart payloads
            validation.error("hgm2_chart.invalid_mapping", f"invalid chart mapping: {exc}", f"charts[{idx}]")
            return None
    validation.error("hgm2_chart.invalid_type", "chart must be ManifoldChart or mapping", f"charts[{idx}]")
    return None


def _validate_hyperedge(edge: Any, idx: int, validation: ValidationResult, config: HGMConfig) -> None:
    if not isinstance(edge, BoundScenarioHyperedge):
        validation.error("hgm2_hyperedge.invalid_type", "hyperedge must be BoundScenarioHyperedge", f"hyperedges[{idx}]")
        return
    if not edge.hyperedge_id:
        validation.error("hgm2_hyperedge.missing_id", "hyperedge_id is required", f"hyperedges[{idx}].hyperedge_id")
    if not finite_number(edge.coherence_score):
        validation.error("hgm2_hyperedge.invalid_coherence", "coherence score must be finite", f"hyperedges[{idx}].coherence_score")
    if not finite_number(edge.probability_score):
        validation.error("hgm2_hyperedge.invalid_probability", "probability score must be finite", f"hyperedges[{idx}].probability_score")


def _validate_charts(raw_charts: Sequence[Any], config: HGMConfig) -> Tuple[Tuple[ManifoldChart, ...], ValidationResult]:
    validation = ValidationResult()
    charts: List[ManifoldChart] = []
    seen = set()
    for idx, raw in enumerate(raw_charts):
        chart = _coerce_chart(raw, idx, validation)
        if chart is None:
            continue
        validation.merge(chart.validate(config))
        if chart.chart_id in seen:
            validation.error("hgm2_chart.duplicate_id", f"duplicate chart_id {chart.chart_id!r}", f"charts[{idx}].chart_id")
        seen.add(chart.chart_id)
        charts.append(chart)
    return tuple(charts), validation


def _depth_from_hint(value: Any, default: DepthLayer) -> DepthLayer:
    try:
        return DepthLayer.coerce(value)
    except Exception:
        return default


def _canonical_depth(edge: BoundScenarioHyperedge, input_obj: ManifoldRoutingInput, options: ManifoldRoutingOptions) -> DepthLayer:
    if edge.hyperedge_id in input_obj.depth_hints:
        return _depth_from_hint(input_obj.depth_hints[edge.hyperedge_id], options.default_depth)
    meta = dict(edge.metadata or {})
    for key in ("depth_layer", "depth", "preferred_depth"):
        if key in meta:
            return _depth_from_hint(meta[key], options.default_depth)
    kind_value = getattr(edge.kind, "value", str(edge.kind))
    if kind_value == "procedural":
        return DepthLayer.D5_PROCEDURAL
    if kind_value in {"causal", "conflict"}:
        return DepthLayer.D4_CAUSAL
    if kind_value == "opportunity":
        return DepthLayer.D7_STRATEGIC
    return options.default_depth


def _geometry_preference(edge: BoundScenarioHyperedge, input_obj: ManifoldRoutingInput) -> Optional[GeometryType]:
    pref = input_obj.geometry_preferences.get(edge.hyperedge_id)
    if pref is None:
        pref = input_obj.geometry_preferences.get("default")
    if pref is None:
        meta = dict(edge.metadata or {})
        pref = meta.get("geometry") or meta.get("geometry_type") or meta.get("preferred_geometry")
    if pref is None:
        return None
    try:
        return GeometryType.coerce(pref)
    except Exception:
        return None


def _point_lookup(entity_id: str, chart_id: str, input_obj: ManifoldRoutingInput, chart: ManifoldChart, edge: BoundScenarioHyperedge):
    coords = dict(input_obj.coordinates or {})
    for key in (
        (entity_id, chart_id),
        f"{entity_id}|{chart_id}",
        entity_id,
        chart_id,
    ):
        try:
            if key in coords:
                return coords[key]
        except TypeError:
            pass
    edge_meta = dict(edge.metadata or {})
    chart_meta = dict(chart.metadata or {})
    if "coordinates" in edge_meta:
        return edge_meta["coordinates"]
    if "point" in edge_meta:
        return edge_meta["point"]
    if "origin" in chart_meta:
        return chart_meta["origin"]
    return None


def _chart_anchor(chart: ManifoldChart, input_obj: ManifoldRoutingInput):
    coords = dict(input_obj.coordinates or {})
    for key in (f"chart:{chart.chart_id}", chart.chart_id):
        if key in coords:
            return coords[key]
    meta = dict(chart.metadata or {})
    return meta.get("anchor") or meta.get("origin")


def _fallback_assignment(
    edge: BoundScenarioHyperedge,
    chart: ManifoldChart,
    depth: DepthLayer,
    reason: str,
    idx: int,
    traces: List[TraceRecord],
) -> ManifoldRouteAssignment:
    trace = TraceRecord.create(
        TraceEventKind.VALIDATE,
        "manifold_router.route_hyperedges_to_manifold_charts",
        severity=ValidationSeverity.WARNING,
        payload={"hyperedge_id": edge.hyperedge_id, "chart_id": chart.chart_id, "reason": reason},
    )
    traces.append(trace)
    base_conf = _clamp01((float(edge.coherence_score) + float(edge.probability_score)) / 2.0) if finite_number(edge.coherence_score) and finite_number(edge.probability_score) else 0.5
    return ManifoldRouteAssignment(
        assignment_id=f"hgm2_route_{idx:04d}",
        hyperedge_id=edge.hyperedge_id,
        chart_id=chart.chart_id,
        geometry_type=chart.geometry,
        depth_layer=depth,
        distance_score=0.0,
        similarity_score=0.5,
        confidence=_clamp01(base_conf * 0.75),
        trace_id=trace.trace_id,
        metadata={"routing_mode": "missing_coordinates_fallback", "reason": reason},
    )


def route_hyperedges_to_manifold_charts(
    hyperedges,
    charts=None,
    config: HGMConfig = HGMConfig(),
    routing_options: Optional[ManifoldRoutingOptions | Mapping[str, Any]] = None,
) -> ManifoldRoutingResult:
    """Assign HGM-1 bound hyperedges to compatible manifold charts."""

    input_obj, options = _coerce_input(hyperedges, charts, routing_options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    edge_list = list(input_obj.bound_hyperedges)
    chart_list, chart_validation = _validate_charts(input_obj.available_manifold_charts, config)
    validation.merge(chart_validation)

    for idx, edge in enumerate(edge_list):
        _validate_hyperedge(edge, idx, validation, config)

    if not edge_list:
        validation.warning("hgm2_routing.empty_hyperedges", "empty hyperedge list; returning no assignments", "hyperedges")
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "manifold_router.route_hyperedges_to_manifold_charts",
            severity=ValidationSeverity.WARNING,
            payload={"reason": "empty_hyperedges"},
        ))
        return ManifoldRoutingResult(tuple(), validation, tuple(traces), metadata={"hyperedge_count": 0, "chart_count": len(chart_list)})

    if not chart_list:
        validation.error("hgm2_routing.empty_charts", "at least one manifold chart is required", "charts")

    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "manifold_router.route_hyperedges_to_manifold_charts",
            severity=ValidationSeverity.ERROR,
            payload={"reason": "pre_validation_failed", "hyperedge_count": len(edge_list), "chart_count": len(chart_list)},
        ))
        return ManifoldRoutingResult(tuple(), validation, tuple(traces), metadata={"hyperedge_count": len(edge_list), "chart_count": len(chart_list)})

    assignments: List[ManifoldRouteAssignment] = []
    for edge in sorted(edge_list, key=lambda item: item.hyperedge_id):
        pref = _geometry_preference(edge, input_obj)
        depth = _canonical_depth(edge, input_obj, options)
        candidates = []
        for chart in chart_list:
            if pref is not None and chart.geometry != pref:
                # Still allow non-preferred charts, but with no preference bonus.
                pref_bonus = 0.0
            else:
                pref_bonus = 0.25 if pref is not None else 0.0

            point_a = _point_lookup(edge.hyperedge_id, chart.chart_id, input_obj, chart, edge)
            point_b = _chart_anchor(chart, input_obj)
            if point_a is None or point_b is None:
                if not options.allow_missing_coordinates:
                    continue
                # Coordinates absent: stable fallback route candidate.
                base = _clamp01((float(edge.coherence_score) + float(edge.probability_score)) / 2.0)
                route_score = 0.5 + pref_bonus + 0.25 * base
                candidates.append((route_score, 0.5, 0.0, chart, None, "missing_coordinates"))
                continue

            dist = compute_geometry_distance(point_a, point_b, chart.geometry, config=config)
            if not dist.validation.ok:
                if options.allow_unsupported_fallback:
                    fallback = compute_geometry_distance(point_a, point_b, options.fallback_geometry, config=config)
                    if fallback.validation.ok:
                        dist = fallback
                    else:
                        continue
                else:
                    continue
            base = _clamp01((float(edge.coherence_score) + float(edge.probability_score)) / 2.0)
            route_score = float(dist.similarity) + pref_bonus + 0.25 * base
            candidates.append((route_score, dist.similarity, dist.distance, chart, dist, "distance"))

        if not candidates:
            validation.error("hgm2_routing.no_compatible_chart", f"no compatible chart for hyperedge {edge.hyperedge_id!r}", edge.hyperedge_id)
            traces.append(TraceRecord.create(
                TraceEventKind.FAIL,
                "manifold_router.route_hyperedges_to_manifold_charts",
                severity=ValidationSeverity.ERROR,
                payload={"hyperedge_id": edge.hyperedge_id, "reason": "no_compatible_chart"},
            ))
            continue

        # Stable tie handling: highest route score, then chart_id.
        candidates.sort(key=lambda item: (-float(item[0]), item[3].chart_id, edge.hyperedge_id))
        score, similarity, distance, chart, dist_result, mode = candidates[0]
        if mode == "missing_coordinates":
            assignment = _fallback_assignment(edge, chart, depth, "missing_coordinates", len(assignments), traces)
        else:
            trace = TraceRecord.create(
                TraceEventKind.CREATE,
                "manifold_router.route_hyperedges_to_manifold_charts",
                payload={"hyperedge_id": edge.hyperedge_id, "chart_id": chart.chart_id, "geometry": chart.geometry.value},
            )
            traces.append(trace)
            assignment = ManifoldRouteAssignment(
                assignment_id=f"hgm2_route_{len(assignments):04d}",
                hyperedge_id=edge.hyperedge_id,
                chart_id=chart.chart_id,
                geometry_type=chart.geometry,
                depth_layer=depth,
                distance_score=float(distance),
                similarity_score=_clamp01(float(similarity)),
                confidence=_clamp01(score / 1.5),
                trace_id=trace.trace_id,
                metadata={"routing_mode": "distance", "preference_geometry": pref.value if pref else None},
            )
        assignments.append(assignment)

    if not validation.ok:
        return ManifoldRoutingResult(tuple(assignments), validation, tuple(traces), metadata={"hyperedge_count": len(edge_list), "chart_count": len(chart_list)})

    traces.append(TraceRecord.create(
        TraceEventKind.VALIDATE,
        "manifold_router.route_hyperedges_to_manifold_charts",
        payload={"assignment_count": len(assignments), "hyperedge_count": len(edge_list)},
    ))
    return ManifoldRoutingResult(
        tuple(assignments),
        validation,
        tuple(traces),
        metadata={"hyperedge_count": len(edge_list), "chart_count": len(chart_list), "assignment_count": len(assignments)},
    )


def build_hgm2_manifold_routing(
    hyperedges,
    charts=None,
    config: HGMConfig = HGMConfig(),
    routing_options: Optional[ManifoldRoutingOptions | Mapping[str, Any]] = None,
):
    """High-level HGM-2 entry point: route hyperedges and build depth targets."""

    from .depth_retrieval import assign_depth_retrieval_targets
    from .hgm2_result import HGM2ManifoldRoutingResult

    routing = route_hyperedges_to_manifold_charts(hyperedges, charts, config=config, routing_options=routing_options)
    depth_bridge = assign_depth_retrieval_targets(routing.assignments, config=config)
    validation = ValidationResult.combine([routing.validation, depth_bridge.validation])
    traces = tuple(routing.trace_records) + tuple(depth_bridge.trace_records)
    if not validation.ok:
        traces = traces + (TraceRecord.create(
            TraceEventKind.FAIL,
            "manifold_router.build_hgm2_manifold_routing",
            severity=ValidationSeverity.ERROR,
            payload={"reason": "hgm2_validation_failed"},
        ),)
    else:
        traces = traces + (TraceRecord.create(
            TraceEventKind.VALIDATE,
            "manifold_router.build_hgm2_manifold_routing",
            payload={"assignment_count": len(routing.assignments), "target_count": len(depth_bridge.targets)},
        ),)
    return HGM2ManifoldRoutingResult(
        routing=routing,
        depth_bridge=depth_bridge,
        validation=validation,
        trace_records=traces,
        metadata={
            "assignment_count": len(routing.assignments),
            "target_count": len(depth_bridge.targets),
            "lineage": ("HGM-0A", "HGM-0B", "HGM-1", "HGM-2"),
        },
    )
