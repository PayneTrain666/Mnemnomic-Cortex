import torch

from mnemonic_cortex.working_memory import (
    QuaternionDepthReplicator,
    WMIntraDepthTransformerConfig,
    WMIntraDepthTransformer,
    WMCrossDepthTransformerConfig,
    WMCrossDepthTransformer,
)


def test_intra_depth_transformer_preserves_shape_and_trace():
    x = torch.randn(2, 8, 5, 3, 32)
    module = WMIntraDepthTransformer(
        WMIntraDepthTransformerConfig(dim=32, num_depths=8, num_heads=4, num_layers=1)
    )
    y, trace = module(x, return_trace=True)

    assert y.shape == x.shape
    assert trace["input_shape"] == list(x.shape)
    assert trace["output_shape"] == list(x.shape)
    assert trace["finite"] is True
    assert trace["stream_count"] == 2 * 8 * 3
    assert "paamax_metadata" in trace
    assert torch.isfinite(y).all()


def test_cross_depth_transformer_preserves_shape_and_trace():
    x = torch.randn(2, 8, 5, 3, 32)
    module = WMCrossDepthTransformer(
        WMCrossDepthTransformerConfig(dim=32, num_depths=8, num_heads=4, num_layers=1)
    )
    y, trace = module(x, return_trace=True)

    assert y.shape == x.shape
    assert trace["input_shape"] == list(x.shape)
    assert trace["output_shape"] == list(x.shape)
    assert trace["finite"] is True
    assert trace["depth_sequence_count"] == 2 * 5 * 3
    assert len(trace["depth_energy"]) == 8
    assert torch.isfinite(y).all()


def test_quaternion_replicator_intra_cross_pipeline():
    rep = QuaternionDepthReplicator(dim=32, num_depths=8)
    intra = WMIntraDepthTransformer(WMIntraDepthTransformerConfig(dim=32, num_depths=8, num_heads=4))
    cross = WMCrossDepthTransformer(WMCrossDepthTransformerConfig(dim=32, num_depths=8, num_heads=4))

    x = torch.randn(2, 5, 32)
    depth = rep(x)
    y1 = intra(depth)
    y2, trace = cross(y1, return_trace=True)

    assert depth.shape == (2, 8, 5, 3, 32)
    assert y1.shape == depth.shape
    assert y2.shape == depth.shape
    assert trace["finite"] is True
    assert torch.isfinite(y2).all()


def test_transformers_reject_bad_shape():
    intra = WMIntraDepthTransformer(WMIntraDepthTransformerConfig(dim=32, num_depths=8, num_heads=4))
    cross = WMCrossDepthTransformer(WMCrossDepthTransformerConfig(dim=32, num_depths=8, num_heads=4))
    bad = torch.randn(2, 5, 32)

    for module in (intra, cross):
        try:
            module(bad)
        except ValueError:
            continue
        raise AssertionError("Expected ValueError for bad shape")
