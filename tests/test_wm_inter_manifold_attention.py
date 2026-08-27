import torch

from mnemonic_cortex.working_memory import (
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    WMInterManifoldAttention,
    WMInterManifoldAttentionConfig,
)
from geometry.chart_native import retract_from_origin, tangent_at_origin


def test_inter_manifold_native_chart_identity_at_zero_mix():
    module = WMInterManifoldAttention(
        WMInterManifoldAttentionConfig(dim=32, num_heads=4, residual_mix=0.2, enable_native_chart_attention=True)
    )
    tokens = torch.randn(2, 5, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    y = module(tokens, depth_state=depth_state, context_map_name="hierarchical", residual_mix=0.0)
    assert torch.allclose(y, tokens)


def test_inter_manifold_native_chart_scores_and_transports():
    native = WMInterManifoldAttention(
        WMInterManifoldAttentionConfig(dim=32, num_heads=4, residual_mix=0.2, enable_native_chart_attention=True)
    )
    ambient = WMInterManifoldAttention(
        WMInterManifoldAttentionConfig(dim=32, num_heads=4, residual_mix=0.2, enable_native_chart_attention=False)
    )
    ambient.load_state_dict(native.state_dict())
    tokens = torch.randn(2, 5, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    views = {"hg": torch.randn(2, 5, 32), "mann": torch.randn(2, 3, 32)}
    y_native, packed = native(
        tokens,
        depth_state=depth_state,
        context_map_name="hierarchical",
        system_views=views,
        return_trace=True,
    )
    y_ambient = ambient(
        tokens,
        depth_state=depth_state,
        context_map_name="hierarchical",
        system_views=views,
    )
    assert y_native.shape == tokens.shape
    assert torch.isfinite(y_native).all()
    assert packed["trace"]["payload"]["native_chart_attention"] is True
    assert native.last_stats["native_chart_attention"] == 1.0
    assert not torch.allclose(y_native, y_ambient)
    weights = native.last_output.attention_weights
    assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.size(0), weights.size(1)), atol=1e-4)


def test_inter_manifold_euclidean_tangent_is_identity():
    x = torch.randn(4, 16)
    t = tangent_at_origin(x, "euclidean")
    y = retract_from_origin(t, "euclidean")
    assert torch.allclose(t, x)
    assert torch.allclose(y, x)


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
    assert payload["native_chart_attention"] is True
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
