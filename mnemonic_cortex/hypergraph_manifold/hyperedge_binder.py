"""Scenario candidate binding for HGM-1."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .coherence import score_hyperedge_coherence
from .config import HGMConfig
from .enums import HyperedgeKind, TraceEventKind, ValidationSeverity
from .hgm1_result import (
    BoundScenarioHyperedge,
    HyperedgeBindingInput,
    HyperedgeBindingOptions,
    HyperedgeBindingResult,
    HGM1ScenarioGraphResult,
)
from .runtime_result import ScenarioCandidate
from .shapes import finite_number
from .types import TraceRecord
from .validation import ValidationResult

_ALLOWED_GROUP_ATTRS = {"variable_id", "magnitude_bin_id", "depth", "context_id", "time_index", "action_id"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _candidate_node_id(candidate: ScenarioCandidate) -> str:
    depth = "na" if candidate.depth is None else str(candidate.depth)
    ctx = "na" if candidate.context_id is None else str(candidate.context_id)
    time = "na" if candidate.time_index is None else str(candidate.time_index)
    action = "na" if candidate.action_id is None else str(candidate.action_id)
    return f"{candidate.variable_id}|{candidate.magnitude_bin_id}|d={depth}|c={ctx}|t={time}|a={action}"


def _group_key(candidate: ScenarioCandidate, group_by: Sequence[str]) -> Tuple[Any, ...]:
    values = []
    for attr in group_by:
        if attr not in _ALLOWED_GROUP_ATTRS:
            values.append((attr, None))
        else:
            values.append((attr, getattr(candidate, attr)))
    return tuple(values)


def _validate_candidates(candidates: Sequence[ScenarioCandidate], config: HGMConfig) -> ValidationResult:
    result = ValidationResult()
    seen_ids = set()
    for idx, candidate in enumerate(candidates):
        path = f"candidates[{idx}]"
        if not candidate.candidate_id:
            result.error("hgm1_candidate.missing_id", "candidate_id is required", f"{path}.candidate_id")
        if candidate.candidate_id in seen_ids:
            result.error("hgm1_candidate.duplicate_id", f"duplicate candidate_id {candidate.candidate_id!r}", f"{path}.candidate_id")
        seen_ids.add(candidate.candidate_id)
        if not candidate.variable_id:
            result.error("hgm1_candidate.missing_variable", "variable_id is required", f"{path}.variable_id")
        if not candidate.magnitude_bin_id:
            result.error("hgm1_candidate.missing_magnitude", "magnitude_bin_id is required", f"{path}.magnitude_bin_id")
        if not finite_number(candidate.probability) or not 0.0 <= float(candidate.probability) <= 1.0:
            result.error("hgm1_candidate.invalid_probability", "probability must be finite and in [0, 1]", f"{path}.probability")
        if not finite_number(candidate.score):
            result.error("hgm1_candidate.invalid_score", "score must be finite", f"{path}.score")
        if candidate.depth is not None and not 0 <= int(candidate.depth) < config.max_depth_layers:
            result.error("hgm1_candidate.invalid_depth", "depth must be within configured depth layers", f"{path}.depth")
    return result


def _coerce_input(candidates_or_input: Iterable[ScenarioCandidate] | HyperedgeBindingInput) -> HyperedgeBindingInput:
    if isinstance(candidates_or_input, HyperedgeBindingInput):
        return candidates_or_input
    return HyperedgeBindingInput(tuple(candidates_or_input))


def _coerce_options(binding_options: Optional[HyperedgeBindingOptions | Mapping[str, Any]]) -> HyperedgeBindingOptions:
    if binding_options is None:
        return HyperedgeBindingOptions()
    if isinstance(binding_options, HyperedgeBindingOptions):
        return binding_options
    return HyperedgeBindingOptions(**dict(binding_options))


def bind_scenario_candidates(
    candidates,
    config: HGMConfig = HGMConfig(),
    binding_options: Optional[HyperedgeBindingOptions | Mapping[str, Any]] = None,
) -> HyperedgeBindingResult:
    """Convert HGM-0B ScenarioCandidate records into bounded hyperedges."""

    input_obj = _coerce_input(candidates)
    options = _coerce_options(binding_options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    candidate_list = list(input_obj.candidates)

    validation.merge(_validate_candidates(candidate_list, config))
    invalid_group_attrs = tuple(attr for attr in options.group_by if attr not in _ALLOWED_GROUP_ATTRS)
    if invalid_group_attrs:
        validation.error("hgm1_binding.unsupported_group_by", f"unsupported grouping attributes: {invalid_group_attrs}", "binding_options.group_by")

    if not candidate_list:
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "hyperedge_binder.bind_scenario_candidates",
            severity=ValidationSeverity.WARNING,
            payload={"reason": "empty_candidates"},
        ))
        validation.warning("hgm1_binding.empty_candidates", "empty candidates; returning no hyperedges", "candidates")
        return HyperedgeBindingResult(tuple(), validation, tuple(traces), metadata={"candidate_count": 0})

    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "hyperedge_binder.bind_scenario_candidates",
            severity=ValidationSeverity.ERROR,
            payload={"candidate_count": len(candidate_list), "reason": "candidate_validation_failed"},
        ))
        return HyperedgeBindingResult(tuple(), validation, tuple(traces), metadata={"candidate_count": len(candidate_list)})

    max_size = options.max_hyperedge_size or config.max_hyperedge_nodes
    max_size = max(1, min(int(max_size), config.max_hyperedge_nodes))
    min_size = 1 if (options.include_singletons or config.allow_singleton_hyperedges) else 2

    groups: Dict[Tuple[Any, ...], List[ScenarioCandidate]] = {}
    for candidate in candidate_list:
        groups.setdefault(_group_key(candidate, options.group_by), []).append(candidate)

    hyperedges: List[BoundScenarioHyperedge] = []
    skipped_singletons = 0
    split_count = 0
    for group_idx, (group, members) in enumerate(sorted(groups.items(), key=lambda item: repr(item[0]))):
        # Stable member ordering: high score/probability first, then candidate_id.
        ordered = sorted(members, key=lambda c: (-float(c.score), -float(c.probability), c.candidate_id))
        for chunk_idx in range(0, len(ordered), max_size):
            chunk = ordered[chunk_idx:chunk_idx + max_size]
            if len(chunk) < min_size:
                skipped_singletons += 1
                continue
            if chunk_idx > 0:
                split_count += 1
            node_ids = tuple(_candidate_node_id(c) for c in chunk)
            if len(set(node_ids)) != len(node_ids):
                validation.error("hgm1_binding.duplicate_nodes", "candidate chunk produced duplicate node IDs", f"groups[{group_idx}]")
                continue
            prob = sum(float(c.probability) for c in chunk) / len(chunk)
            trace = TraceRecord.create(
                TraceEventKind.CREATE,
                "hyperedge_binder.bind_scenario_candidates",
                payload={"group": repr(group), "candidate_count": len(chunk), "max_size": max_size},
            )
            traces.append(trace)
            hyperedge = BoundScenarioHyperedge(
                hyperedge_id=f"hgm1_edge_{len(hyperedges):04d}",
                candidate_ids=tuple(c.candidate_id for c in chunk),
                node_ids=node_ids,
                kind=options.hyperedge_kind,
                coherence_score=0.0,
                probability_score=_clamp01(prob),
                conflict_score=0.0,
                opportunity_score=0.0,
                source_candidate_indices=tuple(c.source_indices for c in chunk),
                trace_id=trace.trace_id,
                metadata={"group_key": group, "chunk_index": chunk_idx // max_size},
            )
            report = score_hyperedge_coherence(hyperedge, candidate_list, config, variable_relation_hints=input_obj.variable_relation_hints)
            for warning in report.warnings:
                validation.warning("hgm1_coherence.warning", warning, hyperedge.hyperedge_id)
            for error in report.errors:
                validation.error("hgm1_coherence.error", error, hyperedge.hyperedge_id)
            hyperedges.append(replace(hyperedge, coherence_score=_clamp01(report.score)))

    if skipped_singletons:
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "hyperedge_binder.bind_scenario_candidates",
            severity=ValidationSeverity.WARNING,
            payload={"skipped_singleton_groups": skipped_singletons},
        ))
    if split_count:
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "hyperedge_binder.bind_scenario_candidates",
            payload={"split_chunks": split_count, "max_hyperedge_size": max_size},
        ))

    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "hyperedge_binder.bind_scenario_candidates",
            severity=ValidationSeverity.ERROR,
            payload={"reason": "binding_validation_failed"},
        ))
        return HyperedgeBindingResult(tuple(), validation, tuple(traces), metadata={"candidate_count": len(candidate_list)})

    traces.append(TraceRecord.create(
        TraceEventKind.VALIDATE,
        "hyperedge_binder.bind_scenario_candidates",
        payload={"candidate_count": len(candidate_list), "hyperedge_count": len(hyperedges)},
    ))
    return HyperedgeBindingResult(
        tuple(hyperedges),
        validation,
        tuple(traces),
        metadata={
            "candidate_count": len(candidate_list),
            "hyperedge_count": len(hyperedges),
            "group_by": options.group_by,
            "max_hyperedge_size": max_size,
            "skipped_singletons": skipped_singletons,
        },
    )


def build_hgm1_scenario_graph(
    candidates,
    config: HGMConfig = HGMConfig(),
    binding_options: Optional[HyperedgeBindingOptions | Mapping[str, Any]] = None,
) -> HGM1ScenarioGraphResult:
    """High-level HGM-1 entry point."""

    from .conflict_graph import detect_conflict_edges
    from .opportunity_graph import detect_opportunity_edges

    binding = bind_scenario_candidates(candidates, config, binding_options)
    input_obj = _coerce_input(candidates)
    conflict = detect_conflict_edges(binding.hyperedges, config, candidates=input_obj.candidates)
    opportunity = detect_opportunity_edges(binding.hyperedges, config, candidates=input_obj.candidates)
    validation = ValidationResult.combine([binding.validation, conflict.validation, opportunity.validation])
    traces = tuple(binding.trace_records) + tuple(conflict.trace_records) + tuple(opportunity.trace_records)
    return HGM1ScenarioGraphResult(
        binding=binding,
        conflict_graph=conflict,
        opportunity_graph=opportunity,
        validation=validation,
        trace_records=traces,
        metadata={
            "hyperedge_count": len(binding.hyperedges),
            "conflict_edge_count": len(conflict.conflict_edges),
            "opportunity_edge_count": len(opportunity.opportunity_edges),
            "lineage_tags": config.lineage_tags,
        },
    )
