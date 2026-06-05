from mnemonic_cortex.hypergraph_manifold import (
    HGMConfig,
    HyperedgeBindingInput,
    HyperedgeBindingOptions,
    ScenarioCandidate,
    bind_scenario_candidates,
    build_hgm1_scenario_graph,
    detect_conflict_edges,
    detect_opportunity_edges,
    score_hyperedge_coherence,
)


def candidate(cid, variable, mag, prob=0.5, *, depth=1, context="ctx", time=0, action="act", metadata=None):
    return ScenarioCandidate(
        candidate_id=cid,
        variable_id=variable,
        magnitude_bin_id=mag,
        probability=prob,
        score=prob,
        source_indices=(0, 0),
        trace_id=f"trace_{cid}",
        depth=depth,
        context_id=context,
        time_index=time,
        action_id=action,
        metadata=metadata or {},
    )


def test_valid_candidates_bind_into_scenario_hyperedges():
    candidates = [candidate("c1", "grip", "up", 0.8), candidate("c2", "slip", "down", 0.7)]
    result = bind_scenario_candidates(candidates)
    assert result.validation.ok
    assert len(result.hyperedges) == 1
    edge = result.hyperedges[0]
    assert edge.candidate_ids == ("c1", "c2")
    assert edge.kind.value == "scenario"
    assert edge.coherence_score > 0.0
    assert edge.probability_score == 0.75


def test_empty_candidates_return_structured_empty_result():
    result = bind_scenario_candidates([])
    assert result.validation.ok
    assert result.hyperedges == tuple()
    assert any(msg.code == "hgm1_binding.empty_candidates" for msg in result.validation.warnings)
    assert result.trace_records


def test_duplicate_candidate_ids_fail_closed():
    candidates = [candidate("dup", "grip", "up"), candidate("dup", "slip", "down")]
    result = bind_scenario_candidates(candidates)
    assert not result.validation.ok
    assert result.hyperedges == tuple()
    assert any(msg.code == "hgm1_candidate.duplicate_id" for msg in result.validation.errors)


def test_bounded_hyperedge_size_is_enforced():
    candidates = [
        candidate("c1", "v1", "up", 0.9),
        candidate("c2", "v2", "up", 0.8),
        candidate("c3", "v3", "up", 0.7),
    ]
    result = bind_scenario_candidates(candidates, binding_options=HyperedgeBindingOptions(max_hyperedge_size=2, include_singletons=True))
    assert result.validation.ok
    assert len(result.hyperedges) == 2
    assert [len(edge.candidate_ids) for edge in result.hyperedges] == [2, 1]


def test_coherence_scoring_is_deterministic():
    candidates = [candidate("c1", "grip", "up", 0.8), candidate("c2", "slip", "down", 0.7)]
    result = bind_scenario_candidates(candidates)
    edge = result.hyperedges[0]
    report1 = score_hyperedge_coherence(edge, candidates)
    report2 = score_hyperedge_coherence(edge, candidates)
    assert report1.score == report2.score
    assert report1.method == "deterministic_heuristic_v1"


def test_conflict_graph_detects_incompatible_positive_negative_candidates():
    candidates = [
        candidate("c1", "grip", "up", 0.8, metadata={"direction": "positive"}),
        candidate("c2", "grip", "down", 0.7, metadata={"direction": "negative"}),
    ]
    binding = bind_scenario_candidates(candidates)
    conflict = detect_conflict_edges(binding.hyperedges, candidates=candidates)
    assert conflict.validation.ok
    assert len(conflict.conflict_edges) == 1
    assert conflict.conflict_edges[0].variable_id == "grip"
    assert conflict.conflict_score > 0.0


def test_opportunity_graph_detects_beneficial_convergence_candidates():
    candidates = [
        candidate("c1", "heat", "recover", 0.8, metadata={"opportunity": True}),
        candidate("c2", "cold", "recover", 0.7, metadata={"utility": 0.9}),
    ]
    binding = bind_scenario_candidates(candidates)
    opportunity = detect_opportunity_edges(binding.hyperedges, candidates=candidates)
    assert opportunity.validation.ok
    assert len(opportunity.opportunity_edges) == 1
    assert opportunity.opportunity_score > 0.0


def test_trace_records_are_generated_and_redacted():
    candidates = [
        candidate("c1", "grip", "up", 0.8),
        candidate("c2", "slip", "down", 0.7),
    ]
    result = build_hgm1_scenario_graph(candidates)
    assert result.trace_records
    trace = result.trace_records[0]
    # TraceRecord redaction remains inherited from HGM-0A.
    hacked = type(trace)(
        trace_id=trace.trace_id,
        event_kind=trace.event_kind,
        component=trace.component,
        timestamp_utc=trace.timestamp_utc,
        severity=trace.severity,
        lineage=trace.lineage,
        parent_trace_id=trace.parent_trace_id,
        payload={"api_key": "abc", "safe": "ok"},
    )
    assert hacked.redacted_payload()["api_key"] == "<redacted>"
    assert hacked.redacted_payload()["safe"] == "ok"


def test_missing_metadata_degrades_safely():
    candidates = [
        candidate("c1", "grip", "up", 0.8, depth=None, context=None, time=None, action=None),
        candidate("c2", "slip", "down", 0.7, depth=None, context=None, time=None, action=None),
    ]
    result = bind_scenario_candidates(candidates)
    assert result.validation.ok
    assert len(result.hyperedges) == 1
    assert any(msg.code == "hgm1_coherence.warning" for msg in result.validation.warnings)


def test_hgm0b_scenario_candidate_records_remain_compatible():
    candidates = [candidate("scenario_0000_0_0", "v0", "m0", 0.5), candidate("scenario_0001_1_1", "v1", "m1", 0.4)]
    result = build_hgm1_scenario_graph(candidates)
    assert result.validation.ok
    assert len(result.binding.hyperedges) == 1
    assert result.metadata["hyperedge_count"] == 1


def test_binding_input_relation_hints_are_supported():
    candidates = [candidate("c1", "grip", "up", 0.5), candidate("c2", "slip", "down", 0.5)]
    binding_input = HyperedgeBindingInput(candidates=tuple(candidates), variable_relation_hints={("grip", "slip"): 1.0})
    result = bind_scenario_candidates(binding_input)
    assert result.validation.ok
    assert result.hyperedges[0].coherence_score > 0.4


def test_invalid_group_by_fails_closed():
    candidates = [candidate("c1", "grip", "up"), candidate("c2", "slip", "down")]
    result = bind_scenario_candidates(candidates, binding_options={"group_by": ("bad_axis",)})
    assert not result.validation.ok
    assert any(msg.code == "hgm1_binding.unsupported_group_by" for msg in result.validation.errors)
