import json
import torch
import pytest

from mnemonic_cortex.working_memory import (
    WMAttentionValidationError,
    ensure_attention_query,
    ensure_candidate_tensor,
    ensure_attention_scores,
    stable_softmax,
    bounded_attention_topk,
    ensure_lane_output,
    attention_trace,
    attention_contract_trace,
    summarize_attention_tensor,
)


def test_attention_query_candidate_and_score_validation():
    q2 = torch.randn(2, 32)
    q3 = torch.randn(2, 5, 32)
    c = torch.randn(2, 4, 32)
    scores = torch.randn(2, 4)

    assert ensure_attention_query("q2", q2, expected_dim=32) is q2
    assert ensure_attention_query("q3", q3, expected_dim=32) is q3
    assert ensure_candidate_tensor("c", c, expected_dim=32) is c
    assert ensure_attention_scores("scores", scores) is scores

    bad = q2.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(WMAttentionValidationError):
        ensure_attention_query("bad", bad)


def test_stable_softmax_and_bounded_topk_are_finite_and_bounded():
    scores = torch.tensor([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    weights = stable_softmax(scores, dim=-1)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)

    vals, idx = bounded_attention_topk(scores, k=99, dim=-1)
    assert vals.shape[-1] == 3
    assert idx.shape[-1] == 3


def test_lane_output_validation_requires_trace_metadata():
    output = {
        "candidate_vectors": torch.randn(2, 4, 32),
        "scores": torch.randn(2, 4),
        "trace": {"ok": True},
    }
    ensure_lane_output("lane", output, expected_dim=32)

    with pytest.raises(WMAttentionValidationError):
        ensure_lane_output("bad_lane", {"candidate_vectors": torch.randn(2, 4, 32)}, expected_dim=32)


def test_attention_trace_and_summary_are_json_safe():
    trace = attention_trace(module="unit", message="ok", lane="vector", payload={"scores": torch.randn(2, 4)})
    contract = attention_contract_trace(module="unit")
    summary = summarize_attention_tensor("x", torch.randn(2, 3))

    json.dumps(trace)
    json.dumps(contract)
    json.dumps(summary)
    assert trace["paamax_metadata"]["attention_contract"] is True
    assert contract["payload"]["bounded_topk_required"] is True
