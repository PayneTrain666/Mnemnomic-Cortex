"""Conflict graph detection for HGM-1."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .config import HGMConfig
from .enums import TraceEventKind, ValidationSeverity
from .hgm1_result import BoundScenarioHyperedge, ConflictEdge, ConflictGraphResult
from .runtime_result import ScenarioCandidate
from .types import TraceRecord
from .validation import ValidationResult


def _candidate_map(candidates: Iterable[ScenarioCandidate]) -> Dict[str, ScenarioCandidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _direction(candidate: ScenarioCandidate) -> str:
    raw = candidate.metadata.get("direction") if candidate.metadata else None
    if raw is not None:
        text = str(raw).strip().lower()
        if text in {"positive", "pos", "+", "increase", "up"}:
            return "positive"
        if text in {"negative", "neg", "-", "decrease", "down"}:
            return "negative"
        if text in {"neutral", "zero", "none"}:
            return "neutral"
    mag = str(candidate.magnitude_bin_id).lower()
    if any(token in mag for token in ("negative", "neg", "down", "decrease", "minus", "-")):
        return "negative"
    if any(token in mag for token in ("positive", "pos", "up", "increase", "plus", "+")):
        return "positive"
    return "unknown"


def detect_conflict_edges(
    hyperedges: Iterable[BoundScenarioHyperedge],
    config: HGMConfig = HGMConfig(),
    *,
    candidates: Iterable[ScenarioCandidate] = tuple(),
) -> ConflictGraphResult:
    """Detect contradictory positive/negative mutation bundles.

    HGM-1 detects conflicts when a hyperedge contains positive and negative
    mutation directions for the same variable. Direction can be explicit in
    candidate metadata or inferred from magnitude-bin naming.
    """

    validation = ValidationResult()
    traces: List[TraceRecord] = []
    cmap = _candidate_map(candidates)
    conflicts: List[ConflictEdge] = []

    for hyperedge in hyperedges:
        members = [cmap[cid] for cid in hyperedge.candidate_ids if cid in cmap]
        if not members:
            validation.warning("hgm1_conflict.no_candidate_metadata", "no candidates available for conflict detection", hyperedge.hyperedge_id)
            continue
        by_variable: Dict[str, Dict[str, List[ScenarioCandidate]]] = {}
        for candidate in members:
            by_variable.setdefault(candidate.variable_id, {}).setdefault(_direction(candidate), []).append(candidate)
        for variable_id, direction_map in sorted(by_variable.items()):
            positives = direction_map.get("positive", [])
            negatives = direction_map.get("negative", [])
            if positives and negatives:
                for left in positives:
                    for right in negatives:
                        score = min(1.0, (float(left.probability) + float(right.probability)) / 2.0)
                        trace = TraceRecord.create(
                            TraceEventKind.CREATE,
                            "conflict_graph.detect_conflict_edges",
                            severity=ValidationSeverity.WARNING,
                            payload={
                                "hyperedge_id": hyperedge.hyperedge_id,
                                "variable_id": variable_id,
                                "left": left.candidate_id,
                                "right": right.candidate_id,
                            },
                        )
                        traces.append(trace)
                        conflicts.append(ConflictEdge(
                            conflict_id=f"conflict_{len(conflicts):04d}",
                            left_candidate_id=left.candidate_id,
                            right_candidate_id=right.candidate_id,
                            variable_id=variable_id,
                            reason="opposing mutation directions for same variable",
                            score=score,
                            trace_id=trace.trace_id,
                            metadata={"hyperedge_id": hyperedge.hyperedge_id},
                        ))

    if not conflicts:
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "conflict_graph.detect_conflict_edges",
            payload={"conflict_edges": 0, "reason": "no_conflicts_detected"},
        ))
    conflict_score = max((edge.score for edge in conflicts), default=0.0)
    return ConflictGraphResult(
        conflict_edges=tuple(conflicts),
        conflict_score=conflict_score,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"conflict_edge_count": len(conflicts)},
    )
