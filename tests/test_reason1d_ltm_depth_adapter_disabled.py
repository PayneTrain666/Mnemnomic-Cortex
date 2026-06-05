import json
import torch

from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig


def test_reason1d_ltm_depth_adapter_disabled_passes_through():
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.disabled(key_dim=16))
    query = torch.randn(2, 5, 16)
    out, trace = adapter.read_ltm(query, bank_name="cgmn_semantic", return_trace=True)
    assert out is query
    assert trace["enabled"] is False
    assert trace["pass_through"] is True
    assert trace["bank_name"] == "cgmn_semantic"
    assert trace["paamax_metadata"]["no_memory_store_mutation"] is True
    json.dumps(trace)
