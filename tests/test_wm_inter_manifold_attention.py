import torch

from mnemonic_cortex.working_memory import (
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    WMInterManifoldAttention,
    WMInterManifoldAttentionConfig,
)


def test_inter_manifold_attention_identity_at_zero_mix():
    module = WMInterManifoldAttention(WMInterManifoldAttentionConfig(dim=32, num_heads=4, residual_mix=0.15))
    tokens = torch.randn(2, 5, 32)
    y = module(tokens, residual_mix=0.0)
    assert y.shape == tokens.shape
    assert torch.allclose(y, tokens)


def test_inter_manifold_attention_monitors_geometry_and_mann_views():
    module = WMInterManifoldAttention(WMInterManifoldAttentionConfig(dim=32, num_heads=4, residual_mix=0.2))
    tokens = torch.randn(2, 5, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    views = {
        "hg": torch.randn(2, 5, 32),
        "cgmn": torch.randn(2, 5, 32),
        "mann": torch.randn(2, 3, 32),
        "psls": torch.randn(2, 4, 32),
    }
    y, packed = module(
        tokens,
        depth_state=depth_state,
        context_map_name="hierarchical",
        system_views=views,
        geometry_weights={"hg": {"hyperbolic": 0.8, "euclidean": 0.2}},
        return_trace=True,
    )

    assert y.shape == tokens.shape
    assert torch.isfinite(y).all()
    payload = packed["trace"]["payload"]
    assert payload["qspin_live_routing"] is False
    assert payload["shared_slot_writes"] is False
    assert payload["ltm_writes"] is False
    assert payload["mann_writes"] is False
    assert payload["qh_writes"] is False
    assert payload["token_count"] >= 8
    assert payload["top_edges"]
    assert any(label.startswith("wm:") for label in packed["labels"])
    assert any(label.startswith("mann:") for label in packed["labels"])
    assert module.last_stats["enabled"] == 1.0


def test_inter_manifold_attention_does_not_shadow_module_apply():
    module = WMInterManifoldAttention(WMInterManifoldAttentionConfig(dim=32, num_heads=4))
    seen = []

    def _mark(mod):
        seen.append(type(mod).__name__)

    module.apply(_mark)
    assert "WMInterManifoldAttention" in seen
    assert hasattr(module, "mix_manifold_communications")


def test_qdt_working_memory_uses_inter_manifold_attention():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    y, trace = wm(x, operation="read", context_map_name="hierarchical", return_trace=True)
    stages = {item["stage"] for item in trace["items"]}
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert "inter_manifold_attention" in stages
    assert wm.last_inter_manifold_stats["token_count"] >= 8
    metrics = wm.get_metrics()
    assert metrics["ima_enabled"] == 1.0


def test_inter_manifold_attention_stability_report():
    module = WMInterManifoldAttention(WMInterManifoldAttentionConfig(dim=16, num_heads=2))
    report = module.stability_report(torch.randn(1, 4, 16))
    assert report["ok"] is True
    assert report["qspin_live_routing"] is False
