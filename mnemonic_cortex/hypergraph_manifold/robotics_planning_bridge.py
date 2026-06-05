"""Advisory robotics planning bridge for HGM-3.

This module emits typed action-plan options only. It does not call robot
hardware, issue actuator commands, or integrate with live controllers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .action_sequence import build_action_sequence_from_hgm2_route
from .enums import TraceEventKind, ValidationSeverity
from .hgm1_result import BoundScenarioHyperedge
from .hgm2_result import DepthRetrievalTarget, HGM2ManifoldRoutingResult, ManifoldRouteAssignment
from .hgm3_result import (
    HGM3ProceduralMemoryResult,
    ProceduralActionSequence,
    ProceduralMemoryRetrievalCandidate,
    ProceduralMemoryRetrievalResult,
    ProceduralMemoryStoreResult,
    RoboticsPlanningActionOption,
    RoboticsPlanningBridgeResult,
    SPCPProceduralOptions,
)
from .procedural_memory import retrieve_similar_procedures, store_procedural_sequences
from .types import TraceRecord
from .validation import ValidationResult


def _coerce_options(options: Optional[SPCPProceduralOptions | Mapping[str, Any]]) -> SPCPProceduralOptions:
    if options is None:
        return SPCPProceduralOptions()
    if isinstance(options, SPCPProceduralOptions):
        return options
    return SPCPProceduralOptions(**dict(options))


def _trace(component: str, validation: ValidationResult, payload=None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload=dict(payload or {}),
    )


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def build_robotics_planning_options(
    retrieval_candidates: Sequence[ProceduralMemoryRetrievalCandidate],
    sequence_lookup: Mapping[str, ProceduralActionSequence],
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> RoboticsPlanningBridgeResult:
    """Convert retrieval candidates into advisory robotics planning options."""

    validation = ValidationResult()
    traces: List[TraceRecord] = []
    candidates = tuple(retrieval_candidates or tuple())
    lookup = dict(sequence_lookup or {})
    if not candidates:
        validation.warning("hgm3_planning.empty_candidates", "empty retrieval candidates; no planning options generated", "retrieval_candidates")
        trace = _trace("robotics_planning_bridge.build_robotics_planning_options", validation, {"reason": "empty_candidates"})
        traces.append(trace)
        return RoboticsPlanningBridgeResult(tuple(), validation, tuple(traces), metadata={"advisory_only": True})

    out: List[RoboticsPlanningActionOption] = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, ProceduralMemoryRetrievalCandidate):
            validation.error("hgm3_planning.invalid_candidate", "candidate must be ProceduralMemoryRetrievalCandidate", f"retrieval_candidates[{idx}]")
            continue
        seq = lookup.get(candidate.sequence_id)
        if seq is None:
            validation.warning("hgm3_planning.missing_sequence", f"sequence {candidate.sequence_id!r} missing from lookup", f"retrieval_candidates[{idx}]")
            continue
        risk = _clamp01(1.0 - (0.70 * candidate.confidence + 0.30 * seq.confidence))
        confidence = _clamp01(0.60 * candidate.confidence + 0.40 * seq.confidence)
        trace = _trace(
            "robotics_planning_bridge.build_robotics_planning_options",
            validation,
            {"candidate_id": candidate.candidate_id, "sequence_id": candidate.sequence_id, "advisory_only": True},
        )
        traces.append(trace)
        out.append(RoboticsPlanningActionOption(
            option_id=f"hgm3_plan_{idx:04d}_{candidate.sequence_id}",
            sequence_id=candidate.sequence_id,
            action_primitives=seq.primitives,
            expected_goal=seq.goal_label,
            risk_score=risk,
            confidence=confidence,
            explanation=(
                "Advisory procedural option retrieved by SPCP similarity; "
                "not an actuator command and not suitable for direct hardware execution."
            ),
            trace_id=trace.trace_id,
            metadata={
                "retrieval_candidate_id": candidate.candidate_id,
                "similarity": candidate.similarity,
                "distance": candidate.distance,
                "advisory_only": True,
                "actuator_execution": False,
            },
        ))
    out.sort(key=lambda item: (-item.confidence, item.risk_score, item.sequence_id, item.option_id))
    final_trace = _trace("robotics_planning_bridge.build_robotics_planning_options", validation, {"option_count": len(out)})
    traces.append(final_trace)
    return RoboticsPlanningBridgeResult(tuple(out), validation, tuple(traces), metadata={"option_count": len(out), "advisory_only": True})


def _records_from_hgm2_payload(payload: Any):
    """Extract hyperedges, assignments, targets and optional query from supported payloads."""
    if isinstance(payload, Mapping):
        return (
            tuple(payload.get("hyperedges") or tuple()),
            tuple(payload.get("assignments") or tuple()),
            tuple(payload.get("depth_targets") or payload.get("targets") or tuple()),
            payload.get("query_sequence") or payload.get("query"),
        )
    if isinstance(payload, HGM2ManifoldRoutingResult):
        # HGM-2 result carries assignments and targets, but not source hyperedges.
        return (tuple(), tuple(payload.routing.assignments), tuple(payload.depth_bridge.targets), None)
    if isinstance(payload, tuple) and len(payload) >= 3:
        return (tuple(payload[0] or tuple()), tuple(payload[1] or tuple()), tuple(payload[2] or tuple()), payload[3] if len(payload) > 3 else None)
    return (tuple(), tuple(), tuple(), None)


def build_hgm3_spcp_procedural_memory(
    hgm2_result_or_records,
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> HGM3ProceduralMemoryResult:
    """High-level HGM-3 procedural-memory entry point."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    hyperedges, assignments, targets, query = _records_from_hgm2_payload(hgm2_result_or_records)

    if not hyperedges:
        validation.warning("hgm3.empty_hyperedges", "no source hyperedges supplied; returning structured empty procedural result", "hyperedges")
    if not assignments:
        validation.warning("hgm3.empty_assignments", "no route assignments supplied; returning structured empty procedural result", "assignments")

    assignment_by_edge: Dict[str, ManifoldRouteAssignment] = {getattr(a, "hyperedge_id", ""): a for a in assignments if isinstance(a, ManifoldRouteAssignment)}
    target_by_edge: Dict[str, DepthRetrievalTarget] = {getattr(t, "hyperedge_id", ""): t for t in targets if isinstance(t, DepthRetrievalTarget)}

    sequences: List[ProceduralActionSequence] = []
    for edge in hyperedges:
        if not isinstance(edge, BoundScenarioHyperedge):
            validation.error("hgm3.invalid_hyperedge", "hyperedges must contain BoundScenarioHyperedge records", "hyperedges")
            continue
        build = build_action_sequence_from_hgm2_route(edge, assignment_by_edge.get(edge.hyperedge_id), target_by_edge.get(edge.hyperedge_id), config=config, options=opts)
        validation.merge(build.validation)
        traces.extend(build.trace_records)
        if build.sequence is not None:
            sequences.append(build.sequence)

    store = store_procedural_sequences(sequences, config=config, options=opts)
    validation.merge(store.validation)
    traces.extend(store.trace_records)

    if query is None and store.stored_sequences:
        query = store.stored_sequences[0]
    if query is None:
        retrieval = ProceduralMemoryRetrievalResult(tuple(), ValidationResult(), tuple(), metadata={"top_k": opts.top_k, "reason": "no_query"})
        retrieval.validation.warning("hgm3_retrieve.no_query", "no query supplied and no stored sequence available", "query")
    else:
        retrieval = retrieve_similar_procedures(query, store.stored_embeddings, opts.top_k, config=config, options=opts)
    validation.merge(retrieval.validation)
    traces.extend(retrieval.trace_records)

    lookup = {seq.sequence_id: seq for seq in store.stored_sequences}
    planning = build_robotics_planning_options(retrieval.candidates, lookup, config=config, options=opts)
    validation.merge(planning.validation)
    traces.extend(planning.trace_records)

    final_trace = _trace("robotics_planning_bridge.build_hgm3_spcp_procedural_memory", validation, {
        "sequence_count": len(sequences),
        "stored_count": len(store.stored_sequences),
        "planning_options": len(planning.action_options),
    })
    traces.append(final_trace)
    return HGM3ProceduralMemoryResult(store, retrieval, planning, validation, tuple(traces), metadata={
        "sequence_count": len(sequences),
        "stored_count": len(store.stored_sequences),
        "advisory_only": True,
    })
