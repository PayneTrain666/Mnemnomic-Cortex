import math

from mnemonic_cortex.hypergraph_manifold import (
    ActionPrimitive,
    BoundScenarioHyperedge,
    DepthLayer,
    DepthRetrievalTarget,
    GeometryType,
    HyperedgeKind,
    ManifoldRouteAssignment,
    ProceduralActionSequence,
    SPCPProcedureEmbedding,
    SPCPProceduralOptions,
    build_action_sequence_from_hgm2_route,
    build_hgm3_spcp_procedural_memory,
    build_robotics_planning_options,
    compute_spcp_procedure_embedding,
    retrieve_similar_procedures,
    spcp_procedure_similarity,
    store_procedural_sequences,
    validate_action_primitive,
    validate_action_sequence,
)


def edge(edge_id="edge_proc", **metadata):
    return BoundScenarioHyperedge(
        hyperedge_id=edge_id,
        candidate_ids=(f"{edge_id}_c1", f"{edge_id}_c2"),
        node_ids=(f"{edge_id}_n1", f"{edge_id}_n2"),
        kind=HyperedgeKind.PROCEDURAL,
        coherence_score=0.82,
        probability_score=0.74,
        conflict_score=0.0,
        opportunity_score=0.0,
        source_candidate_indices=((0, 0), (1, 0)),
        trace_id=f"trace_{edge_id}",
        metadata=metadata,
    )


def assignment(edge_id="edge_proc", assignment_id=None):
    return ManifoldRouteAssignment(
        assignment_id=assignment_id or f"assign_{edge_id}",
        hyperedge_id=edge_id,
        chart_id=f"chart_{edge_id}",
        geometry_type=GeometryType.SPCP,
        depth_layer=DepthLayer.D5_PROCEDURAL,
        distance_score=0.1,
        similarity_score=0.9,
        confidence=0.8,
        trace_id=f"trace_assign_{edge_id}",
        metadata={},
    )


def depth_target(edge_id="edge_proc"):
    return DepthRetrievalTarget(
        target_id=f"target_{edge_id}",
        hyperedge_id=edge_id,
        depth_layer=DepthLayer.D5_PROCEDURAL,
        retrieval_key=f"D5_PROCEDURAL:{edge_id}",
        priority=0.8,
        trace_id=f"trace_depth_{edge_id}",
        metadata={},
    )


def primitive(primitive_id="prim_a", **kwargs):
    params = kwargs.pop("parameters", {"x": 1.0, "y": [0.1, 0.2]})
    return ActionPrimitive(
        primitive_id=primitive_id,
        action_type=kwargs.pop("action_type", "move"),
        parameters=params,
        duration=kwargs.pop("duration", 0.5),
        confidence=kwargs.pop("confidence", 0.9),
        frame_id=kwargs.pop("frame_id", "generic_frame"),
        metadata=kwargs,
    )


def sequence(sequence_id="seq_a", primitives=None):
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=tuple(primitives or (primitive("p1"), primitive("p2", action_type="grip"))),
        source_hyperedge_id="edge_proc",
        source_assignment_id="assign_edge_proc",
        source_depth_target_id="target_edge_proc",
        goal_label="test_goal",
        confidence=0.85,
        trace_id="trace_sequence",
        metadata={},
    )


def test_valid_hgm2_route_builds_procedural_action_sequence():
    result = build_action_sequence_from_hgm2_route(edge(), assignment(), depth_target())
    assert result.validation.ok
    assert result.sequence is not None
    assert result.sequence.source_hyperedge_id == "edge_proc"
    assert len(result.sequence.primitives) == 2


def test_empty_inputs_return_structured_empty_result():
    result = build_hgm3_spcp_procedural_memory({"hyperedges": [], "assignments": [], "depth_targets": []})
    assert result.validation.ok
    assert not result.store_result.stored_sequences
    assert result.trace_records


def test_invalid_primitive_fails_closed():
    bad = ActionPrimitive(primitive_id="", action_type="", parameters={"x": float("inf")}, duration=-1.0, confidence=2.0)
    result = validate_action_primitive(bad)
    assert not result.ok
    assert result.errors


def test_bounded_action_sequence_length_is_enforced():
    opts = SPCPProceduralOptions(max_sequence_length=1)
    result = validate_action_sequence(sequence(primitives=(primitive("p1"), primitive("p2"))), opts)
    assert not result.ok
    assert any(m.code == "hgm3_sequence.too_long" for m in result.errors)


def test_spherical_state_is_normalized():
    emb = compute_spcp_procedure_embedding(sequence())
    assert emb.validation.ok
    assert emb.embedding is not None
    norm = math.sqrt(sum(x * x for x in emb.embedding.spherical_state))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_zero_spherical_state_fails_closed():
    zero_seq = ProceduralActionSequence(
        sequence_id="zero_seq",
        primitives=tuple(),
        source_hyperedge_id="edge_zero",
        source_assignment_id="assign_zero",
        source_depth_target_id="target_zero",
        goal_label="",
        confidence=0.0,
        trace_id="trace_zero",
    )
    emb = compute_spcp_procedure_embedding(zero_seq, options=SPCPProceduralOptions(embedding_dimension=1))
    assert not emb.validation.ok
    assert emb.embedding is None


def test_complex_projective_similarity_is_phase_insensitive_enough():
    emb_a = compute_spcp_procedure_embedding(sequence("seq_phase_a")).embedding
    assert emb_a is not None
    # Apply global phase i to projective component: (x+iy) -> (-y + ix)
    rotated = []
    vals = emb_a.projective_state
    for idx in range(0, len(vals), 2):
        x, y = vals[idx], vals[idx + 1]
        rotated.extend([-y, x])
    emb_b = SPCPProcedureEmbedding(
        embedding_id="phase_rotated",
        sequence_id="seq_phase_b",
        spherical_state=emb_a.spherical_state,
        projective_state=tuple(rotated),
        conformal_warp=emb_a.conformal_warp,
        similarity_ready=True,
        trace_id="trace_phase_b",
    )
    sim = spcp_procedure_similarity(emb_a, emb_b)
    assert sim.validation.ok
    assert sim.similarity > 0.99


def test_conformal_warp_is_bounded():
    opts = SPCPProceduralOptions(conformal_warp_bound=0.1)
    emb = compute_spcp_procedure_embedding(sequence(), options=opts)
    assert emb.validation.ok
    assert emb.embedding is not None
    assert max(abs(x) for x in emb.embedding.conformal_warp) <= 0.1 + 1e-12


def test_procedural_sequences_store_successfully():
    store = store_procedural_sequences([sequence("seq_store")])
    assert store.validation.ok
    assert len(store.stored_sequences) == 1
    assert len(store.stored_embeddings) == 1


def test_similar_procedures_retrieve_deterministically():
    s1 = sequence("seq_a")
    s2 = sequence("seq_b")
    store = store_procedural_sequences([s2, s1])
    retrieval = retrieve_similar_procedures(s1, store.stored_embeddings, 2)
    assert retrieval.validation.ok
    assert len(retrieval.candidates) == 2
    assert retrieval.candidates[0].similarity >= retrieval.candidates[1].similarity


def test_top_k_non_positive_returns_structured_empty_retrieval_result():
    store = store_procedural_sequences([sequence("seq_topk")])
    retrieval = retrieve_similar_procedures(sequence("seq_topk"), store.stored_embeddings, 0)
    assert retrieval.validation.ok
    assert not retrieval.candidates
    assert retrieval.validation.warnings


def test_robotics_planning_options_are_generated():
    s = sequence("seq_plan")
    store = store_procedural_sequences([s])
    retrieval = retrieve_similar_procedures(s, store.stored_embeddings, 1)
    lookup = {s.sequence_id: s}
    plan = build_robotics_planning_options(retrieval.candidates, lookup)
    assert plan.validation.ok
    assert len(plan.action_options) == 1
    assert plan.action_options[0].metadata["actuator_execution"] is False
    assert "not an actuator command" in plan.action_options[0].explanation


def test_missing_kinematic_metadata_degrades_safely():
    result = build_action_sequence_from_hgm2_route(edge("edge_missing_meta"), assignment("edge_missing_meta"), depth_target("edge_missing_meta"))
    assert result.validation.ok
    assert result.sequence is not None
    assert result.validation.warnings
    assert result.sequence.primitives[0].metadata["generated"] is True


def test_trace_records_are_generated_and_redacted():
    result = build_action_sequence_from_hgm2_route(
        edge("edge_trace", action_primitives=[{"primitive_id": "x", "action_type": "move", "parameters": {"secret_token": "abc"}}]),
        assignment("edge_trace"),
        depth_target("edge_trace"),
    )
    assert result.trace_records
    assert isinstance(result.trace_records[0].redacted_payload(), dict)


def test_high_level_hgm3_result_builds_store_retrieval_and_planning():
    e = edge("edge_high")
    a = assignment("edge_high")
    d = depth_target("edge_high")
    result = build_hgm3_spcp_procedural_memory({"hyperedges": [e], "assignments": [a], "depth_targets": [d]})
    assert result.validation.ok
    assert result.store_result.stored_sequences
    assert result.retrieval_result.candidates
    assert result.planning_bridge_result.action_options
