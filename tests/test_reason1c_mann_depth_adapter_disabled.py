import json
import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig


def test_reason1c_mann_depth_adapter_disabled_passes_through():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.disabled(key_dim=16))
    query = torch.randn(2, 5, 16)
    out, trace = adapter.read_hop(query, hop_id=0, return_trace=True)
    assert out is query
    assert trace["enabled"] is False
    assert trace["pass_through"] is True
    assert trace["hop_id"] == 0
    assert trace["paamax_metadata"]["no_memory_store_mutation"] is True
    json.dumps(trace)
