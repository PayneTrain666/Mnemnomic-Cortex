import torch

from mnemonic_cortex.working_memory import (
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    RetrievalLaneConfig,
    WMRetrievalLanes,
    WMGeometryScoringConfig,
    WMGeometryScoring,
    WMGeometryLinkerConfig,
    WMGeometryLinker,
)


def make_bank(dim=32, slots=8):
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=slots, dim=dim))
    with torch.no_grad():
        bank.importance.fill_(0.6)
        bank.confidence.fill_(0.8)
    bank.trace_links[0].append("trace-a")
    bank.trace_links[1].extend(["trace-b", "trace-c"])
    return bank


def test_retrieval_lanes_all_lane_shapes_and_policy_metadata():
    bank = make_bank()
    lanes = WMRetrievalLanes(RetrievalLaneConfig(dim=32, top_k=3), slot_bank=bank)
    query = torch.randn(2, 32)
    out, trace = lanes(query, return_trace=True)

    expected = {"vector", "hyperbolic", "temporal", "spatial", "procedural", "trace", "policy"}
    assert expected.issubset(set(out.lane_outputs))
    assert trace["paamax_metadata"]["policy_lane_present"] is True
    for lane_name, lane_out in out.lane_outputs.items():
        assert lane_out.candidates.shape == (2, 3, 32)
        assert lane_out.scores.shape == (2, 3)
        assert lane_out.slot_indices.shape == (2, 3)
        assert lane_out.metadata["finite"] is True


def test_geometry_scoring_fuses_candidates():
    bank = make_bank()
    lanes = WMRetrievalLanes(RetrievalLaneConfig(dim=32, top_k=3), slot_bank=bank)
    scoring = WMGeometryScoring(WMGeometryScoringConfig(dim=32, top_k=3))
    query = torch.randn(2, 32)

    retrieval = lanes(query)
    out, trace = scoring(query, retrieval, return_trace=True)

    assert out.candidates.shape[0] == 2
    assert out.candidates.shape[-1] == 32
    assert out.fused_context.shape == (2, 32)
    assert out.weights.shape == out.scores.shape
    assert torch.allclose(out.weights.reshape(2, -1).sum(dim=-1), torch.ones(2), atol=1e-5)
    assert trace["trace"]["paamax_metadata"]["policy_lane_included"] is True


def test_geometry_linker_lane_bias_and_trace():
    linker = WMGeometryLinker(WMGeometryLinkerConfig(dim=32))
    query = torch.randn(2, 32)
    bias = linker.lane_bias(query)
    trace = linker.to_trace()

    assert "vector" in bias
    assert abs(sum(bias.values()) - 1.0) < 1e-5
    assert trace["trace_type"] == "wm_geometry_linker"
    assert len(trace["links"]) >= 1
