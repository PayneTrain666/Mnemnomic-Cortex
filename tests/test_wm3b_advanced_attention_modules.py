import torch

from mnemonic_cortex.working_memory import (
    WMEvidenceAttentionConfig,
    WMEvidenceAttention,
    WMTraceAttentionConfig,
    WMTraceAttention,
    WMCounterfactualAttentionConfig,
    WMCounterfactualAttention,
    WMConflictAttentionConfig,
    WMConflictAttention,
    WMNoveltyAttentionConfig,
    WMNoveltyAttention,
    WMStabilityAttentionConfig,
    WMStabilityAttention,
)


def test_evidence_attention_shape_trace():
    module = WMEvidenceAttention(WMEvidenceAttentionConfig(dim=32, evidence_slots=4))
    tokens = torch.randn(2, 5, 32)
    memory = torch.randn(2, 32)
    y, trace = module(tokens, memory_context=memory, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["evidence_context_shape"] == [2, 32]
    assert trace["trace"]["paamax_metadata"]["trace_type"] == "wm_evidence_attention"
    assert torch.isfinite(y).all()


def test_trace_attention_uses_prior_trace_features():
    module = WMTraceAttention(WMTraceAttentionConfig(dim=32))
    tokens = torch.randn(2, 5, 32)
    prior = {"items": [{"stage": "a"}] * 3, "confidence": 0.7, "disagreement": 0.2, "paamax_metadata": {"policy_lane_present": True}}
    y, trace = module(tokens, prior_trace=prior, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["trace_score_shape"] == [2]
    assert trace["trace"]["paamax_metadata"]["trace_governance"] is True


def test_counterfactual_attention_trace():
    module = WMCounterfactualAttention(WMCounterfactualAttentionConfig(dim=32))
    tokens = torch.randn(2, 5, 32)
    memory = torch.randn(2, 32)
    y, trace = module(tokens, memory_context=memory, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["counterfactual_delta_shape"] == [2, 32]
    assert "harmful_memory_score" in trace["trace"]["paamax_metadata"] or trace["trace"]["paamax_metadata"]["counterfactual_probe"] is True


def test_conflict_attention_quarantine_metadata():
    module = WMConflictAttention(WMConflictAttentionConfig(dim=32, conflict_threshold=0.1))
    tokens = torch.ones(2, 5, 32)
    memory = -torch.ones(2, 32)
    y, trace = module(tokens, memory_context=memory, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["trace"]["paamax_metadata"]["quarantine_required"] is True
    assert any(trace["quarantine_mask"])


def test_novelty_attention_lightbulb_metadata():
    module = WMNoveltyAttention(WMNoveltyAttentionConfig(dim=32, lightbulb_threshold=0.01))
    tokens = torch.randn(2, 5, 32)
    memory = torch.zeros(2, 32)
    y, trace = module(tokens, memory_context=memory, return_trace=True)

    assert y.shape == tokens.shape
    assert "lightbulb" in trace["trace"]["paamax_metadata"]
    assert trace["trace"]["paamax_metadata"]["trace_type"] == "wm_novelty_attention"


def test_stability_attention_repairs_nan_inf():
    module = WMStabilityAttention(WMStabilityAttentionConfig(dim=32, max_norm=10.0))
    tokens = torch.randn(2, 5, 32)
    tokens[0, 0, 0] = float("nan")
    tokens[1, 0, 0] = float("inf")
    y, trace = module(tokens, return_trace=True)

    assert y.shape == tokens.shape
    assert torch.isfinite(y).all()
    assert trace["repaired"] is True
    assert trace["trace"]["paamax_metadata"]["stability_guard"] is True
