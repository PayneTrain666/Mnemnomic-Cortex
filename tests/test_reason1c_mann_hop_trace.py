import json
import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig


def test_reason1c_hop_trace_contains_required_fields():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=18))
    _, trace = adapter.read_hop(torch.randn(2, 16), hop_id=2, return_trace=True)

    for key in [
        "hop_id",
        "selected_slots",
        "selected_depths",
        "depth_entropy",
        "support_mass",
        "confidence",
        "disagreement",
        "canonical_slot_ids",
    ]:
        assert key in trace
    assert trace["hop_id"] == 2
    assert isinstance(trace["selected_slots"], list)
    assert isinstance(trace["selected_depths"], list)
    json.dumps(trace)
