"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: opportunity graph.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Opportunity graph detection for HGM-1.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .config import HGMConfig
from .enums import TraceEventKind
from .hgm1_result import BoundScenarioHyperedge, OpportunityEdge, OpportunityGraphResult
from .runtime_result import ScenarioCandidate
from .types import TraceRecord
from .validation import ValidationResult


def _candidate_map(candidates: Iterable[ScenarioCandidate]) -> Dict[str, ScenarioCandidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _candidate_utility(candidate: ScenarioCandidate) -> float:
    if not candidate.metadata:
        return 0.0
    if bool(candidate.metadata.get("opportunity")) or bool(candidate.metadata.get("beneficial")):
        return 1.0
    value = candidate.metadata.get("utility", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def detect_opportunity_edges(
    hyperedges: Iterable[BoundScenarioHyperedge],
    config: HGMConfig = HGMConfig(),
    *,
    candidates: Iterable[ScenarioCandidate] = tuple(),
) -> OpportunityGraphResult:
    """Detect beneficial convergence bundles.

    A first deterministic HGM-1 opportunity edge is emitted when a hyperedge has
    at least two candidates with positive utility/opportunity metadata and a
    non-trivial coherence score.
    """

    validation = ValidationResult()
    traces: List[TraceRecord] = []
    cmap = _candidate_map(candidates)
    opportunity_edges: List[OpportunityEdge] = []

    for hyperedge in hyperedges:
        members = [cmap[cid] for cid in hyperedge.candidate_ids if cid in cmap]
        useful = [candidate for candidate in members if _candidate_utility(candidate) > 0.0]
        if len(useful) >= 2 and hyperedge.coherence_score >= 0.25:
            utility = sum(_candidate_utility(candidate) for candidate in useful) / len(useful)
            score = max(0.0, min(1.0, (utility + hyperedge.coherence_score + hyperedge.probability_score) / 3.0))
            trace = TraceRecord.create(
                TraceEventKind.CREATE,
                "opportunity_graph.detect_opportunity_edges",
                payload={"hyperedge_id": hyperedge.hyperedge_id, "candidate_count": len(useful), "score": score},
            )
            traces.append(trace)
            opportunity_edges.append(OpportunityEdge(
                opportunity_id=f"opportunity_{len(opportunity_edges):04d}",
                candidate_ids=tuple(candidate.candidate_id for candidate in useful),
                reason="beneficial convergence metadata within coherent hyperedge",
                score=score,
                trace_id=trace.trace_id,
                metadata={"hyperedge_id": hyperedge.hyperedge_id},
            ))

    if not opportunity_edges:
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "opportunity_graph.detect_opportunity_edges",
            payload={"opportunity_edges": 0, "reason": "insufficient_beneficial_convergence"},
        ))
    opportunity_score = max((edge.score for edge in opportunity_edges), default=0.0)
    return OpportunityGraphResult(
        opportunity_edges=tuple(opportunity_edges),
        opportunity_score=opportunity_score,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"opportunity_edge_count": len(opportunity_edges)},
    )
