import torch

from mnemonic_cortex.working_memory import (
    WMTraceEmitter,
    WMTripletProjector,
    WMDepthAdapters,
    WMDepthAdaptersConfig,
    WMDepthFusion,
    WMDepthFusionConfig,
)


def test_wm_trace_emitter_summary_and_dict():
    emitter = WMTraceEmitter()
    trace = emitter.start("read", policy_lane=True)
    trace.add("unit", "hello", value=1)
    trace = emitter.finish(trace, confidence=0.9, disagreement=0.1)

    d = trace.to_dict()
    s = trace.summary()
    assert d["operation"] == "read"
    assert s["item_count"] >= 2
    assert d["confidence"] == 0.9
    assert "policy_lane" in d["paamax_metadata"]


def test_triplet_projector_state_shape_and_fuse():
    projector = WMTripletProjector(dim=32)
    x = torch.randn(2, 5, 32)
    fused, state = projector(x, return_state=True)

    assert fused.shape == x.shape
    assert state.tensor.shape == (2, 5, 3, 32)
    assert state.shape_summary()["tensor"] == [2, 5, 3, 32]


def test_depth_adapters_preserve_shape_and_trace():
    module = WMDepthAdapters(WMDepthAdaptersConfig(dim=32, num_depths=8))
    x = torch.randn(2, 8, 5, 3, 32)
    y, trace = module(x, return_trace=True)

    assert y.shape == x.shape
    assert trace["input_shape"] == list(x.shape)
    assert trace["output_shape"] == list(x.shape)
    assert trace["finite"] is True
    assert torch.isfinite(y).all()


def test_depth_fusion_shape_trace_and_disagreement():
    fusion = WMDepthFusion(WMDepthFusionConfig(dim=32, num_depths=8))
    x = torch.randn(2, 8, 5, 3, 32)
    residual = torch.randn(2, 5, 32)
    y, trace = fusion(x, residual=residual, return_trace=True)

    assert y.shape == (2, 5, 32)
    assert trace["input_shape"] == list(x.shape)
    assert trace["output_shape"] == [2, 5, 32]
    assert trace["finite"] is True
    assert len(trace["depth_weights"]) == 8
    assert len(trace["triplet_weights"]) == 3
    assert trace["disagreement"] >= 0.0
