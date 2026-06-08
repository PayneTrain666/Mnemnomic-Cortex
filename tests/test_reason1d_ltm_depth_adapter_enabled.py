import json
import torch

from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig


def test_reason1d_ltm_depth_adapter_enabled_read_is_finite_and_serializable():
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=16, value_dim=18, slot_count=20))
    query = torch.randn(3, 4, 16)
    out, trace = adapter.read_ltm(query, bank_name="hg_episodic", return_trace=True)

    assert out.shape == (3, 18)
    assert torch.isfinite(out).all()
    assert trace["trace_type"] == "ltm_depth_trace"
    assert trace["bank_name"] == "hg_episodic"
    for field in ["selected_slots", "selected_depths", "depth_entropy", "confidence", "disagreement", "canonical_slot_ids"]:
        assert field in trace
    json.dumps(trace)


def test_reason1d_ltm_depth_adapter_curved_associative_bank():
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=16, value_dim=18, slot_count=20))
    query = torch.randn(2, 16)
    out, trace = adapter.read_ltm(query, bank_name="curved", return_trace=True)

    assert out.shape == (2, 18)
    assert torch.isfinite(out).all()
    assert trace["bank_name"] == "curved_associative"
    metrics = adapter.capacity_metrics()
    assert "curved_associative" in metrics
