"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: depth retrieval.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Depth-layer retrieval bridge for HGM-2.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .config import HGMConfig
from .enums import DepthLayer, TraceEventKind, ValidationSeverity
from .hgm2_result import DepthRetrievalBridgeResult, DepthRetrievalTarget, ManifoldRouteAssignment
from .shapes import finite_number
from .types import TraceRecord
from .validation import ValidationResult

_CANONICAL_DEPTH_LABELS = {
    DepthLayer.D0_OBSERVATION: "observation",
    DepthLayer.D1_MUTATION: "mutation",
    DepthLayer.D2_ENTITY: "entity",
    DepthLayer.D3_RELATION: "relation",
    DepthLayer.D4_CAUSAL: "causal",
    DepthLayer.D5_PROCEDURAL: "procedural",
    DepthLayer.D6_COUNTERFACTUAL: "counterfactual",
    DepthLayer.D7_STRATEGIC: "strategic",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_depth(value: Any, default: DepthLayer) -> DepthLayer:
    try:
        return DepthLayer.coerce(value)
    except Exception:
        return default


def assign_depth_retrieval_targets(
    assignments,
    config: HGMConfig = HGMConfig(),
    depth_options: Optional[Mapping[str, Any]] = None,
) -> DepthRetrievalBridgeResult:
    """Convert manifold route assignments into depth-layer retrieval targets."""

    options = dict(depth_options or {})
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    assignment_list = tuple(assignments or tuple())
    if not assignment_list:
        validation.warning("hgm2_depth.empty_assignments", "empty assignments; returning no retrieval targets", "assignments")
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "depth_retrieval.assign_depth_retrieval_targets",
            severity=ValidationSeverity.WARNING,
            payload={"reason": "empty_assignments"},
        ))
        return DepthRetrievalBridgeResult(tuple(), validation, tuple(traces), metadata={"assignment_count": 0})

    depth_overrides = dict(options.get("depth_overrides", {}))
    targets: List[DepthRetrievalTarget] = []
    for idx, assignment in enumerate(sorted(assignment_list, key=lambda item: (item.depth_layer.value, item.hyperedge_id, item.chart_id))):
        if not isinstance(assignment, ManifoldRouteAssignment):
            validation.error("hgm2_depth.invalid_assignment_type", "assignment must be ManifoldRouteAssignment", f"assignments[{idx}]")
            continue
        if not assignment.hyperedge_id:
            validation.error("hgm2_depth.missing_hyperedge", "hyperedge_id is required", f"assignments[{idx}].hyperedge_id")
        if not assignment.chart_id:
            validation.error("hgm2_depth.missing_chart", "chart_id is required", f"assignments[{idx}].chart_id")
        if not finite_number(assignment.confidence):
            validation.error("hgm2_depth.invalid_confidence", "assignment confidence must be finite", f"assignments[{idx}].confidence")
            priority = 0.0
        else:
            priority = _clamp01(float(assignment.confidence))
        depth = _coerce_depth(depth_overrides.get(assignment.hyperedge_id, assignment.depth_layer), assignment.depth_layer)
        label = _CANONICAL_DEPTH_LABELS.get(depth, "unknown")
        trace = TraceRecord.create(
            TraceEventKind.CREATE,
            "depth_retrieval.assign_depth_retrieval_targets",
            payload={"hyperedge_id": assignment.hyperedge_id, "depth": depth.name, "label": label},
        )
        traces.append(trace)
        retrieval_key = f"{depth.name}:{label}:{assignment.chart_id}:{assignment.hyperedge_id}"
        targets.append(DepthRetrievalTarget(
            target_id=f"hgm2_depth_{len(targets):04d}",
            hyperedge_id=assignment.hyperedge_id,
            depth_layer=depth,
            retrieval_key=retrieval_key,
            priority=priority,
            trace_id=trace.trace_id,
            metadata={
                "chart_id": assignment.chart_id,
                "geometry_type": assignment.geometry_type.value,
                "depth_label": label,
                "assignment_id": assignment.assignment_id,
            },
        ))

    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "depth_retrieval.assign_depth_retrieval_targets",
            severity=ValidationSeverity.ERROR,
            payload={"reason": "depth_validation_failed"},
        ))
        return DepthRetrievalBridgeResult(tuple(), validation, tuple(traces), metadata={"assignment_count": len(assignment_list)})

    traces.append(TraceRecord.create(
        TraceEventKind.VALIDATE,
        "depth_retrieval.assign_depth_retrieval_targets",
        payload={"target_count": len(targets)},
    ))
    return DepthRetrievalBridgeResult(tuple(targets), validation, tuple(traces), metadata={"assignment_count": len(assignment_list), "target_count": len(targets)})
