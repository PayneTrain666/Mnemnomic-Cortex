"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: action sequence.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
HGM-3 action-sequence construction utilities.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence

from .enums import DepthLayer, TraceEventKind, ValidationSeverity
from .hgm1_result import BoundScenarioHyperedge
from .hgm2_result import DepthRetrievalTarget, ManifoldRouteAssignment
from .hgm3_result import (
    ActionPrimitive,
    ActionSequenceBuildResult,
    ProceduralActionSequence,
    SPCPProceduralOptions,
)
from .shapes import finite_number
from .types import TraceRecord
from .validation import ValidationResult


def _coerce_options(options: Optional[SPCPProceduralOptions | Mapping[str, Any]]) -> SPCPProceduralOptions:
    if options is None:
        return SPCPProceduralOptions()
    if isinstance(options, SPCPProceduralOptions):
        return options
    return SPCPProceduralOptions(**dict(options))


def _clamp01(value: float) -> float:
    if not finite_number(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _trace(component: str, severity=ValidationSeverity.INFO, payload=None) -> TraceRecord:
    return TraceRecord.create(TraceEventKind.CREATE, component, severity=severity, payload=dict(payload or {}))


def _numeric_parameter_values(parameters: Mapping[str, Any], validation: ValidationResult, path: str, options: SPCPProceduralOptions) -> None:
    if len(parameters) > options.max_parameter_count:
        validation.error("hgm3_primitive.too_many_parameters", "action parameters exceed max_parameter_count", path)
    for key, value in dict(parameters or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not finite_number(value):
            validation.error("hgm3_primitive.non_finite_parameter", "numeric action parameters must be finite", f"{path}.{key}")
        if isinstance(value, (list, tuple)):
            if len(value) > options.max_parameter_count:
                validation.error("hgm3_primitive.parameter_vector_too_long", "parameter vector exceeds max_parameter_count", f"{path}.{key}")
            for idx, item in enumerate(value):
                if isinstance(item, (int, float)) and not finite_number(item):
                    validation.error("hgm3_primitive.non_finite_parameter", "numeric action parameters must be finite", f"{path}.{key}[{idx}]")


def validate_action_primitive(primitive: Any, index: int = 0, options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None) -> ValidationResult:
    opts = _coerce_options(options)
    result = ValidationResult()
    if not isinstance(primitive, ActionPrimitive):
        result.error("hgm3_primitive.invalid_type", "primitive must be ActionPrimitive", f"primitives[{index}]")
        return result
    if not primitive.primitive_id:
        result.error("hgm3_primitive.missing_id", "primitive_id is required", f"primitives[{index}].primitive_id")
    if not primitive.action_type:
        result.error("hgm3_primitive.missing_action_type", "action_type is required", f"primitives[{index}].action_type")
    if not finite_number(primitive.duration) or float(primitive.duration) < 0.0:
        result.error("hgm3_primitive.invalid_duration", "duration must be finite and non-negative", f"primitives[{index}].duration")
    if not finite_number(primitive.confidence) or not (0.0 <= float(primitive.confidence) <= 1.0):
        result.error("hgm3_primitive.invalid_confidence", "confidence must be finite and in [0, 1]", f"primitives[{index}].confidence")
    _numeric_parameter_values(primitive.parameters or {}, result, f"primitives[{index}].parameters", opts)
    return result


def validate_action_sequence(sequence: Any, options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None) -> ValidationResult:
    opts = _coerce_options(options)
    result = ValidationResult()
    if not isinstance(sequence, ProceduralActionSequence):
        result.error("hgm3_sequence.invalid_type", "sequence must be ProceduralActionSequence", "sequence")
        return result
    if not sequence.sequence_id:
        result.error("hgm3_sequence.missing_id", "sequence_id is required", "sequence_id")
    if not sequence.source_hyperedge_id:
        result.error("hgm3_sequence.missing_source_hyperedge", "source_hyperedge_id is required", "source_hyperedge_id")
    if len(sequence.primitives) == 0:
        result.error("hgm3_sequence.empty_primitives", "sequence must contain at least one primitive", "primitives")
    if len(sequence.primitives) > opts.max_sequence_length:
        result.error("hgm3_sequence.too_long", "action sequence exceeds max_sequence_length", "primitives")
    if not finite_number(sequence.confidence) or not (0.0 <= float(sequence.confidence) <= 1.0):
        result.error("hgm3_sequence.invalid_confidence", "sequence confidence must be in [0, 1]", "confidence")
    seen = set()
    for idx, primitive in enumerate(sequence.primitives):
        result.merge(validate_action_primitive(primitive, idx, opts))
        if isinstance(primitive, ActionPrimitive):
            if primitive.primitive_id in seen:
                result.error("hgm3_sequence.duplicate_primitive", f"duplicate primitive_id {primitive.primitive_id!r}", f"primitives[{idx}].primitive_id")
            seen.add(primitive.primitive_id)
    return result


def _primitive_from_metadata(raw: Mapping[str, Any], index: int, options: SPCPProceduralOptions) -> ActionPrimitive:
    return ActionPrimitive(
        primitive_id=str(raw.get("primitive_id") or raw.get("id") or f"primitive_{index:04d}"),
        action_type=str(raw.get("action_type") or raw.get("type") or "generic_action"),
        parameters=dict(raw.get("parameters") or {}),
        duration=float(raw.get("duration", options.default_duration)),
        confidence=_clamp01(float(raw.get("confidence", options.default_confidence))),
        frame_id=raw.get("frame_id"),
        object_id=raw.get("object_id"),
        tool_id=raw.get("tool_id"),
        metadata=dict(raw.get("metadata") or {}),
    )


def _generated_primitives(hyperedge: BoundScenarioHyperedge, options: SPCPProceduralOptions) -> List[ActionPrimitive]:
    ids = list(hyperedge.candidate_ids or hyperedge.node_ids or (hyperedge.hyperedge_id,))
    ids = ids[: options.max_sequence_length]
    primitives: List[ActionPrimitive] = []
    base_conf = _clamp01((float(hyperedge.coherence_score) + float(hyperedge.probability_score)) / 2.0)
    for idx, item in enumerate(ids):
        primitives.append(ActionPrimitive(
            primitive_id=f"prim_{hyperedge.hyperedge_id}_{idx:04d}",
            action_type="candidate_step",
            parameters={"source_index": idx, "source_id_hash": float(sum(ord(ch) for ch in str(item)) % 997) / 997.0},
            duration=options.default_duration,
            confidence=base_conf,
            frame_id="generic_action_frame",
            metadata={"source_id": item, "generated": True},
        ))
    return primitives


def build_action_sequence_from_hgm2_route(
    hyperedge: Optional[BoundScenarioHyperedge],
    assignment: Optional[ManifoldRouteAssignment],
    depth_target: Optional[DepthRetrievalTarget],
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> ActionSequenceBuildResult:
    """Convert HGM-1/HGM-2 records into a procedural action sequence."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []

    if hyperedge is None:
        validation.warning("hgm3_sequence.missing_hyperedge", "missing hyperedge; returning structured skip result", "hyperedge")
        trace = _trace("action_sequence.build_action_sequence_from_hgm2_route", ValidationSeverity.WARNING, {"reason": "missing_hyperedge"})
        traces.append(trace)
        return ActionSequenceBuildResult(None, validation, tuple(traces), metadata={"skipped": True})
    if not isinstance(hyperedge, BoundScenarioHyperedge):
        validation.error("hgm3_sequence.invalid_hyperedge", "hyperedge must be BoundScenarioHyperedge", "hyperedge")
        trace = _trace("action_sequence.build_action_sequence_from_hgm2_route", ValidationSeverity.ERROR, {"reason": "invalid_hyperedge"})
        traces.append(trace)
        return ActionSequenceBuildResult(None, validation, tuple(traces), metadata={"skipped": True})

    if assignment is None:
        validation.warning("hgm3_sequence.missing_assignment", "missing route assignment; using fallback assignment id", "assignment")
        source_assignment_id = f"assignment_missing_{hyperedge.hyperedge_id}"
    else:
        source_assignment_id = getattr(assignment, "assignment_id", "") or f"assignment_missing_{hyperedge.hyperedge_id}"

    if depth_target is None:
        validation.warning("hgm3_sequence.missing_depth_target", "missing depth target; using deterministic D5 procedural default", "depth_target")
        source_depth_target_id = f"depth_default_{hyperedge.hyperedge_id}"
        depth_layer = DepthLayer.D5_PROCEDURAL
    else:
        source_depth_target_id = getattr(depth_target, "target_id", "") or f"depth_default_{hyperedge.hyperedge_id}"
        try:
            depth_layer = DepthLayer.coerce(getattr(depth_target, "depth_layer", DepthLayer.D5_PROCEDURAL))
        except Exception:
            depth_layer = DepthLayer.D5_PROCEDURAL

    explicit = dict(hyperedge.metadata or {}).get("action_primitives")
    primitives: List[ActionPrimitive]
    if explicit and isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        primitives = [_primitive_from_metadata(dict(item), idx, opts) for idx, item in enumerate(explicit) if isinstance(item, Mapping)]
    else:
        validation.warning("hgm3_sequence.missing_kinematic_metadata", "explicit action metadata missing; generated generic action primitives", "hyperedge.metadata.action_primitives")
        primitives = _generated_primitives(hyperedge, opts)

    trace = _trace(
        "action_sequence.build_action_sequence_from_hgm2_route",
        ValidationSeverity.INFO if validation.ok else ValidationSeverity.WARNING,
        {"hyperedge_id": hyperedge.hyperedge_id, "depth_layer": getattr(depth_layer, "name", str(depth_layer))},
    )
    traces.append(trace)
    seq = ProceduralActionSequence(
        sequence_id=f"hgm3_seq_{hyperedge.hyperedge_id}",
        primitives=tuple(primitives),
        source_hyperedge_id=hyperedge.hyperedge_id,
        source_assignment_id=source_assignment_id,
        source_depth_target_id=source_depth_target_id,
        goal_label=str(dict(hyperedge.metadata or {}).get("goal_label") or f"goal_from_{hyperedge.hyperedge_id}"),
        confidence=_clamp01((float(hyperedge.coherence_score) + float(hyperedge.probability_score)) / 2.0),
        trace_id=trace.trace_id,
        metadata={
            "depth_layer": getattr(depth_layer, "name", str(depth_layer)),
            "advisory_only": True,
            "robotics_control_execution": False,
        },
    )
    validation.merge(validate_action_sequence(seq, opts))
    return ActionSequenceBuildResult(seq if validation.ok else None, validation, tuple(traces), metadata={"primitive_count": len(primitives)})
