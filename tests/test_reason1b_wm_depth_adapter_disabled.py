import json
import torch

from mnemonic_cortex.reasoning_depth import WMDepthAdapter, WMDepthAdapterConfig


def test_reason1b_wm_depth_adapter_disabled_passes_through():
    adapter = WMDepthAdapter(WMDepthAdapterConfig.disabled(input_dim=16))
    x = torch.randn(2, 5, 16)
    y, trace = adapter.process(x, return_trace=True)

    assert y is x
    assert trace["enabled"] is False
    assert trace["pass_through"] is True
    assert trace["output_shape"] == [2, 5, 16]
    assert trace["paamax_metadata"]["no_memory_store_mutation"] is True
    json.dumps(trace)
