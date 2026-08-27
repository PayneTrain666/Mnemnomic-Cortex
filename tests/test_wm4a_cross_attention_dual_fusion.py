import torch

from mnemonic_cortex.working_memory import (
    WMLTMCrossAttentionConfig,
    WMLTMCrossAttention,
    WMMANNCrossAttentionConfig,
    WMMANNCrossAttention,
    WMSPCPCrossAttentionConfig,
    WMSPCPCrossAttention,
    WMDualFusionConfig,
    WMDualFusionController,
)


def test_ltm_cross_attention_shape_trace():
    module = WMLTMCrossAttention(WMLTMCrossAttentionConfig(dim=32, top_k=3))
    tokens = torch.randn(2, 5, 32)
    y, trace = module(tokens, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["memory_context_shape"] == [2, 32]
    assert trace["trace"]["memory_type"] == "ltm"
    assert torch.isfinite(y).all()


def test_mann_cross_attention_trace_visibility():
    module = WMMANNCrossAttention(WMMANNCrossAttentionConfig(dim=32, top_k=4, hops=3))
    tokens = torch.randn(2, 5, 32)
    y, trace = module(tokens, return_trace=True)

    assert y.shape == tokens.shape
    visibility = trace["visibility"]
    assert visibility["pre_fusion_output_shape"] == [2, 5, 32]
    assert visibility["per_hop_attention_shape"] == [2, 3, 4]
    assert visibility["scratchpad_tokens_shape"] == [2, 3, 32]
    assert len(visibility["confidence"]) == 2
    assert len(visibility["disagreement"]) == 2
    assert trace["trace"]["paamax_metadata"]["mann_trace_visible"] is True


def test_spcp_cross_attention_shape_trace():
    module = WMSPCPCrossAttention(WMSPCPCrossAttentionConfig(dim=32, top_k=3))
    tokens = torch.randn(2, 5, 32)
    y, trace = module(tokens, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["memory_context_shape"] == [2, 32]
    assert trace["trace"]["memory_type"] == "spcp"
    assert trace["trace"]["paamax_metadata"]["procedural_memory"] is True


def test_dual_fusion_controller_shape_trace_confidence_disagreement():
    module = WMDualFusionController(WMDualFusionConfig(dim=32, top_k=3))
    tokens = torch.randn(2, 5, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    y, trace = module(tokens, depth_state=depth_state, return_trace=True)

    assert y.shape == tokens.shape
    assert trace["fused_context_shape"] == [2, 32]
    assert trace["trace"]["pre_fusion_outputs"]["ltm"] == [2, 5, 32]
    assert trace["trace"]["pre_fusion_outputs"]["mann"] == [2, 5, 32]
    assert trace["trace"]["pre_fusion_outputs"]["spcp"] == [2, 5, 32]
    assert trace["trace"]["paamax_metadata"]["mann_trace_visible"] is True
    assert len(trace["trace"]["confidence"]) == 2
    assert len(trace["trace"]["disagreement"]) == 2
    assert torch.isfinite(y).all()


def test_dual_fusion_stability_report():
    module = WMDualFusionController(WMDualFusionConfig(dim=32, top_k=3))
    tokens = torch.randn(2, 5, 32)
    report = module.stability_report(tokens)

    assert report["ok"] is True
    assert report["finite"] is True
    assert report["shape_ok"] is True
    assert report["mann_trace_visible"] is True


def test_prefusion_native_chart_identity_at_zero_residual():
    module = WMLTMCrossAttention(
        WMLTMCrossAttentionConfig(dim=32, top_k=3, residual_mix=0.0, enable_native_chart_attention=True)
    )
    tokens = torch.randn(2, 5, 32)
    y = module(tokens, context_map_name="hierarchical")
    assert torch.allclose(y, tokens)


def test_prefusion_native_chart_scores_and_transports():
    native = WMLTMCrossAttention(
        WMLTMCrossAttentionConfig(dim=32, top_k=3, residual_mix=0.2, enable_native_chart_attention=True)
    )
    ambient = WMLTMCrossAttention(
        WMLTMCrossAttentionConfig(dim=32, top_k=3, residual_mix=0.2, enable_native_chart_attention=False)
    )
    ambient.load_state_dict(native.state_dict())
    tokens = torch.randn(2, 5, 32)
    y_native, packed = native(tokens, context_map_name="hierarchical", return_trace=True)
    y_ambient = ambient(tokens, context_map_name="hierarchical")
    assert y_native.shape == tokens.shape
    assert torch.isfinite(y_native).all()
    assert packed["trace"]["native_chart_attention"] is True
    assert packed["trace"]["paamax_metadata"]["native_chart_score_and_mix"] is True
    assert not torch.allclose(y_native, y_ambient)


def test_dual_fusion_native_chart_attention_trace():
    module = WMDualFusionController(
        WMDualFusionConfig(dim=32, top_k=3, enable_native_chart_attention=True)
    )
    tokens = torch.randn(2, 5, 32)
    y, packed = module(tokens, context_map_name="hierarchical", return_trace=True)
    assert y.shape == tokens.shape
    assert torch.isfinite(y).all()
    assert packed["trace"]["native_chart_attention"] is True
    assert packed["ltm_trace"]["trace"]["native_chart_attention"] is True
    assert packed["mann_trace"]["trace"]["native_chart_attention"] is True
    assert packed["spcp_trace"]["trace"]["native_chart_attention"] is True
