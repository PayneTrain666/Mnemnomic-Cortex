import json
import torch

from mnemonic_cortex.reasoning_depth import WMDepthAdapter, WMDepthAdapterConfig


def test_reason1b_wm_depth_adapter_enabled_depth_read_shape_and_trace():
    adapter = WMDepthAdapter(WMDepthAdapterConfig.enabled_default(input_dim=16, value_dim=16, slot_count=12))
    x = torch.randn(3, 4, 16)
    y, trace = adapter.process(x, return_trace=True)

    assert y.shape == (3, 16)
    assert torch.isfinite(y).all()
    assert trace["trace_type"] == "depth_indexed_lattice_trace"
    assert trace["metadata"]["capacity_multiplier"] == 8
    json.dumps(trace)
