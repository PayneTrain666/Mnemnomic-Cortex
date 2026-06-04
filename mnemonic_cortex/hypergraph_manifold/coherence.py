"""Deterministic coherence scoring for HGM-1 scenario hyperedges."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

from .config import HGMConfig
from .hgm1_result import BoundScenarioHyperedge, CoherenceScoreReport
from .runtime_result import ScenarioCandidate
from .shapes import finite_number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def _candidate_map(candidates: Iterable[ScenarioCandidate]) -> Dict[str, ScenarioCandidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def score_hyperedge_coherence(
    hyperedge: BoundScenarioHyperedge,
    candidates,
    config: HGMConfig = HGMConfig(),
    *,
    variable_relation_hints: Mapping[Tuple[str, str], float] | None = None,
) -> CoherenceScoreReport:
    """Score whether grouped candidates plausibly belong together.

    HGM-1 intentionally uses deterministic heuristics first. Later releases can
    replace or augment this with learned geometry-aware scoring.
    """

    cmap = _candidate_map(candidates)
    members = [cmap[cid] for cid in hyperedge.candidate_ids if cid in cmap]
    warnings = []
    errors = []
    if not members:
        return CoherenceScoreReport(0.0, "deterministic_heuristic_v1", 0, errors=("no candidate members found",))

    evidence_count = 0
    score_parts = []

    # Probability evidence: high average candidate confidence supports coherence.
    finite_probs = [float(c.probability) for c in members if finite_number(c.probability)]
    if finite_probs:
        score_parts.append(_clamp01(sum(finite_probs) / len(finite_probs)))
        evidence_count += len(finite_probs)
    else:
        warnings.append("no finite candidate probabilities")

    # Multi-node scenario evidence: hyperedges with 2-4 atoms are more meaningful
    # than singleton or extremely large bundles.
    size = len(members)
    if size >= 2:
        score_parts.append(_clamp01(min(size, 4) / 4.0))
        evidence_count += 1
    else:
        warnings.append("singleton hyperedge has weak coherence evidence")

    # Shared context/time/action/depth supports common-scenario grouping.
    for attr in ("context_id", "time_index", "action_id", "depth"):
        values = [getattr(c, attr) for c in members if getattr(c, attr) is not None]
        if values and len(set(values)) == 1:
            score_parts.append(0.75)
            evidence_count += 1
        elif not values:
            warnings.append(f"missing {attr} metadata")
        else:
            score_parts.append(0.35)
            evidence_count += 1

    # Optional relation hints between variables.
    relation_hints = variable_relation_hints or {}
    if relation_hints and len(members) > 1:
        pair_scores = []
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                key = _pair_key(left.variable_id, right.variable_id)
                if key in relation_hints and finite_number(relation_hints[key]):
                    pair_scores.append(_clamp01(float(relation_hints[key])))
        if pair_scores:
            score_parts.append(sum(pair_scores) / len(pair_scores))
            evidence_count += len(pair_scores)

    if not score_parts:
        return CoherenceScoreReport(0.0, "deterministic_heuristic_v1", evidence_count, warnings=tuple(warnings), errors=tuple(errors))

    # Average with a slight penalty for reported errors; warnings remain non-fatal.
    score = _clamp01(sum(score_parts) / len(score_parts))
    if errors:
        score *= 0.5
    return CoherenceScoreReport(score, "deterministic_heuristic_v1", evidence_count, warnings=tuple(warnings), errors=tuple(errors))
