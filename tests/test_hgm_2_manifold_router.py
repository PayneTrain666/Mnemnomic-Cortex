import math

from mnemonic_cortex.hypergraph_manifold import (
    BoundScenarioHyperedge,
    DepthLayer,
    GeometryType,
    HyperedgeKind,
    ManifoldChart,
    ManifoldRoutingInput,
    ManifoldRoutingOptions,
    assign_depth_retrieval_targets,
    build_hgm2_manifold_routing,
    compute_geometry_distance,
    route_hyperedges_to_manifold_charts,
)


def edge(edge_id="edge_a", **metadata):
    return BoundScenarioHyperedge(
        hyperedge_id=edge_id,
        candidate_ids=(f"{edge_id}_c1", f"{edge_id}_c2"),
        node_ids=(f"{edge_id}_n1", f"{edge_id}_n2"),
        kind=HyperedgeKind.SCENARIO,
        coherence_score=0.8,
        probability_score=0.6,
        conflict_score=0.0,
        opportunity_score=0.0,
        source_candidate_indices=((0, 0), (1, 0)),
        trace_id=f"trace_{edge_id}",
        metadata=metadata,
    )


def chart(chart_id="chart_a", geometry=GeometryType.EUCLIDEAN, dimension=2, **metadata):
    return ManifoldChart(chart_id=chart_id, geometry=geometry, dimension=dimension, metadata=metadata)


def test_valid_hyperedges_route_to_compatible_manifold_charts():
    e = edge("edge_a", preferred_geometry="euclidean")
    c = chart("chart_a", GeometryType.EUCLIDEAN, 2, anchor=[0.0, 0.0])
    result = route_hyperedges_to_manifold_charts(
        [e],
        [c],
        routing_options=ManifoldRoutingOptions(),
    )
    assert result.validation.ok
    assert len(result.assignments) == 1
    assert result.assignments[0].chart_id == "chart_a"
    assert result.assignments[0].hyperedge_id == "edge_a"


def test_empty_hyperedges_return_structured_empty_result():
    result = route_hyperedges_to_manifold_charts([], [chart()])
    assert result.validation.ok
    assert len(result.assignments) == 0
    assert result.validation.warnings
    assert result.trace_records


def test_invalid_chart_fails_closed():
    result = route_hyperedges_to_manifold_charts([edge()], [{"chart_id": "", "geometry": "euclidean", "dimension": 2}])
    assert not result.validation.ok
    assert len(result.assignments) == 0


def test_unsupported_geometry_fails_closed():
    result = compute_geometry_distance([0.0, 0.0], [1.0, 1.0], "not_a_geometry")
    assert not result.validation.ok
    assert result.similarity == 0.0


def test_euclidean_distance_is_deterministic():
    result = compute_geometry_distance([0.0, 0.0], [3.0, 4.0], GeometryType.EUCLIDEAN)
    assert result.validation.ok
    assert result.distance == 5.0
    assert result.similarity == 1.0 / 6.0


def test_spherical_cosine_distance_is_deterministic():
    result = compute_geometry_distance([1.0, 0.0], [0.0, 1.0], GeometryType.SPHERICAL)
    assert result.validation.ok
    assert math.isclose(result.distance, math.pi / 2.0, rel_tol=1e-9)
    assert math.isclose(result.similarity, 0.5, rel_tol=1e-9)


def test_hyperbolic_approximation_returns_finite_for_valid_ball_points():
    result = compute_geometry_distance([0.1, 0.0], [0.2, 0.0], GeometryType.HYPERBOLIC)
    assert result.validation.ok
    assert math.isfinite(result.distance)
    assert 0.0 <= result.similarity <= 1.0


def test_hyperbolic_approximation_fails_closed_for_outside_ball_points():
    result = compute_geometry_distance([1.2, 0.0], [0.2, 0.0], GeometryType.HYPERBOLIC)
    assert not result.validation.ok
    assert result.similarity == 0.0


def test_torus_wrapped_distance_handles_circular_boundary():
    result = compute_geometry_distance([0.99], [0.01], GeometryType.TORUS)
    assert result.validation.ok
    assert result.distance < 0.03


def test_complex_projective_similarity_is_phase_invariant_enough():
    # [1+0i, 0+0i] and i*[1+0i, 0+0i] = [0+1i, 0+0i]
    result = compute_geometry_distance([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], GeometryType.COMPLEX_PROJECTIVE)
    assert result.validation.ok
    assert result.similarity > 0.999
    assert result.distance < 1e-9


def test_deterministic_tie_handling_chooses_stable_chart_id():
    e = edge("edge_tie")
    charts = [chart("chart_b", GeometryType.EUCLIDEAN), chart("chart_a", GeometryType.EUCLIDEAN)]
    result = route_hyperedges_to_manifold_charts([e], charts)
    assert result.validation.ok
    assert result.assignments[0].chart_id == "chart_a"


def test_depth_retrieval_targets_are_generated():
    result = route_hyperedges_to_manifold_charts([edge("edge_depth")], [chart("chart_depth")])
    bridge = assign_depth_retrieval_targets(result.assignments)
    assert bridge.validation.ok
    assert len(bridge.targets) == 1
    assert bridge.targets[0].depth_layer == DepthLayer.D3_RELATION
    assert "D3_RELATION" in bridge.targets[0].retrieval_key


def test_trace_records_are_generated_and_redacted():
    result = build_hgm2_manifold_routing(
        [edge("edge_trace")],
        [chart("chart_trace")],
        routing_options={"allow_missing_coordinates": True},
    )
    assert result.validation.ok
    assert result.trace_records
    sample = result.trace_records[0]
    # Redaction is provided by TraceRecord and should not throw.
    assert isinstance(sample.redacted_payload(), dict)


def test_missing_coordinates_degrade_safely():
    result = route_hyperedges_to_manifold_charts([edge("edge_missing")], [chart("chart_missing")])
    assert result.validation.ok
    assert result.assignments[0].metadata["routing_mode"] == "missing_coordinates_fallback"
    assert result.trace_records


def test_hgm2_input_envelope_with_coordinates_routes_by_distance():
    e = edge("edge_coord")
    c = chart("chart_coord", GeometryType.EUCLIDEAN, 2, anchor=[0.0, 0.0])
    inp = ManifoldRoutingInput(
        bound_hyperedges=(e,),
        available_manifold_charts=(c,),
        coordinates={"edge_coord": [0.0, 0.0], "chart:chart_coord": [0.0, 0.0]},
    )
    result = route_hyperedges_to_manifold_charts(inp)
    assert result.validation.ok
    assert result.assignments[0].metadata["routing_mode"] == "distance"
    assert result.assignments[0].similarity_score == 1.0
