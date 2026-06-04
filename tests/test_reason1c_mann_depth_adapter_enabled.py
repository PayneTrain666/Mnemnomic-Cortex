import json
import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig


def test_reason1c_mann_depth_adapter_enabled_read_is_finite():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=18, slot_count=20))
    query = torch.randn(3, 4, 16)
    out, trace = adapter.read_hop(query, hop_id=1, return_trace=True)
    assert out.shape == (3, 18)
    assert torch.isfinite(out).all()
    assert trace["trace_type"] == "mann_depth_hop_trace"
    assert trace["hop_id"] == 1
    json.dumps(trace)
