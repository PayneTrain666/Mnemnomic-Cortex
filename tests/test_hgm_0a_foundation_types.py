import pytest

from mnemonic_cortex.hypergraph_manifold import (
    DepthLayer,
    GeometryType,
    HyperedgeKind,
    HypersetMatrix,
    MagnitudeBin,
    ManifoldChart,
    MutationDirection,
    MutationToken,
    P_VM,
    P_VMD,
    ProbabilityNormalizationMode,
    QSpinSignature,
    ScenarioHyperedge,
    TraceEventKind,
    TraceRecord,
)


def bins():
    return (
        MagnitudeBin("neg", -1.0, -0.1, "negative"),
        MagnitudeBin("zero", -0.1, 0.1, "neutral"),
        MagnitudeBin("pos", 0.1, 1.0, "positive"),
    )


def test_valid_hyperset_matrix_construction_and_validation():
    matrix = HypersetMatrix(
        matrix_id="hm_valid",
        probabilities=[
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
        ],
        contract=P_VM,
        variable_ids=("grip_force", "object_slip"),
        magnitude_bins=bins(),
        normalization_mode=ProbabilityNormalizationMode.MUTATION_AXIS,
    )
    result = matrix.validate()
    assert result.ok, [m.message for m in result.messages]
    assert matrix.shape == (2, 3)


def test_invalid_probability_rows_are_rejected():
    matrix = HypersetMatrix(
        matrix_id="hm_bad_rows",
        probabilities=[
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.1],
        ],
        contract=P_VM,
        variable_ids=("grip_force", "object_slip"),
        magnitude_bins=bins(),
    )
    result = matrix.validate()
    assert not result.ok
    assert any(m.code == "probability.mutation_axis_not_normalized" for m in result.errors)


def test_invalid_depth_layer_is_rejected_by_coercion():
    with pytest.raises(ValueError):
        DepthLayer.coerce(8)


def test_invalid_geometry_is_rejected_by_coercion():
    with pytest.raises(ValueError):
        ManifoldChart(chart_id="chart_bad", geometry="banana_geometry", dimension=3)


def test_malformed_hyperedge_is_rejected():
    edge = ScenarioHyperedge(
        hyperedge_id="edge_bad",
        node_ids=("single_node",),
        kind=HyperedgeKind.SCENARIO,
        weight=1.0,
    )
    result = edge.validate()
    assert not result.ok
    assert any(m.code == "hyperedge.too_few_nodes" for m in result.errors)


def test_trace_generation_and_validation_redacts_secret_like_payload():
    trace = TraceRecord.create(
        TraceEventKind.CREATE,
        "hgm_0a_test",
        payload={"api_token": "secret-value", "safe": "ok"},
    )
    result = trace.validate()
    assert result.ok
    assert trace.trace_id.startswith("trace_")
    assert trace.redacted_payload()["api_token"] == "<redacted>"
    assert trace.redacted_payload()["safe"] == "ok"


def test_mutation_token_and_qspin_validate():
    token = MutationToken(
        token_id="mut_1",
        variable_id="wrist_angle",
        direction=MutationDirection.POSITIVE,
        magnitude_bin_id="pos",
        probability=0.72,
        depth_layer=DepthLayer.D1_MUTATION,
        geometry=GeometryType.SPHERICAL,
    )
    assert token.validate().ok

    qspin = QSpinSignature("qs_1", components=(0.0,) * 8, phase=0.25)
    assert qspin.validate().ok


def test_vmd_shape_contract_requires_depth_metadata():
    matrix = HypersetMatrix(
        matrix_id="hm_depth",
        probabilities=[
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
        contract=P_VMD,
        variable_ids=("x",),
        magnitude_bins=bins(),
        # depth_layers intentionally omitted
    )
    result = matrix.validate()
    assert not result.ok
    assert any(m.code == "hyperset_matrix.missing_axis_metadata" for m in result.errors)
