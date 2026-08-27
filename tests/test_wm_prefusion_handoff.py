import torch

from geometry.chart_native import tangent_at_origin
from mnemonic_cortex.working_memory import (
    FUSION_SPACE,
    PreFusionHandoff,
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    WMLTMCrossAttention,
    WMLTMCrossAttentionConfig,
    WMDualFusionConfig,
    WMDualFusionController,
)
from mnemonic_cortex.working_memory.wm_prefusion_handoff import wm_token_handoff


def test_default_prefusion_has_no_handoff_payload():
    module = WMLTMCrossAttention(WMLTMCrossAttentionConfig(dim=32, top_k=3))
    tokens = torch.randn(2, 5, 32)
    y, packed = module(tokens, context_map_name="hierarchical", return_trace=True)
    assert y.shape == tokens.shape
    assert "handoff" not in packed
    assert packed["trace"]["prefusion_handoff"] is None
    assert module.last_output.handoff is None


def test_prefusion_handoff_emits_tangent_contract():
    module = WMLTMCrossAttention(
        WMLTMCrossAttentionConfig(dim=32, top_k=3, enable_prefusion_handoff=True)
    )
    tokens = torch.randn(2, 5, 32)
    y, packed = module(tokens, context_map_name="hierarchical", return_trace=True)
    handoff = packed["handoff"]
    assert y.shape == tokens.shape
    assert handoff["space"] == FUSION_SPACE
    assert handoff["system"] == "ltm"
    assert handoff["query_chart"]
    assert torch.equal(module.last_output.handoff.tangent, module.last_output.memory_context)
    assert torch.isfinite(module.last_output.handoff.tangent).all()


def test_wm_token_handoff_is_euclidean_identity():
    tokens = torch.randn(3, 4, 16)
    handoff = wm_token_handoff(tokens, map_name="literal")
    pooled = tokens.mean(dim=1)
    assert isinstance(handoff, PreFusionHandoff)
    assert handoff.space == FUSION_SPACE
    assert torch.allclose(handoff.tangent, pooled)
    assert torch.allclose(handoff.tangent, tangent_at_origin(pooled, "euclidean"))


def test_fusion_policy_links_prefusion_handoff():
    module = WMDualFusionController(
        WMDualFusionConfig(
            dim=32,
            top_k=3,
            enable_chart_fusion_policy=True,
            chart_fusion_condition_mix=0.0,
        )
    )
    assert module.ltm.config.enable_prefusion_handoff is True
    assert module.mann.config.enable_prefusion_handoff is True
    assert module.spcp.config.enable_prefusion_handoff is True
    tokens = torch.randn(2, 5, 32)
    y, packed = module(tokens, context_map_name="procedural", return_trace=True)
    handoff = packed["trace"]["prefusion_handoff"]
    assert y.shape == tokens.shape
    assert handoff["enabled"] is True
    assert handoff["complete"] is True
    assert handoff["space"] == FUSION_SPACE
    for system in ("wm", "ltm", "mann", "spcp"):
        assert handoff["systems"][system]["space"] == FUSION_SPACE
        assert handoff["systems"][system]["tangent_shape"] == [2, 32]


def test_handoff_opt_in_without_fusion_policy():
    module = WMDualFusionController(
        WMDualFusionConfig(dim=32, top_k=3, enable_prefusion_handoff=True)
    )
    assert module.fusion_policy is None
    tokens = torch.randn(2, 5, 32)
    _, packed = module(tokens, context_map_name="hierarchical", return_trace=True)
    assert packed["trace"]["prefusion_handoff"]["complete"] is True
    assert packed["trace"]["chart_fusion_policy"]["enabled"] is False


def test_qdt_handoff_default_off_and_policy_links():
    off = QDTWorkingMemory(
        QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    )
    assert off.get_metrics()["pfh_enabled"] == 0.0
    on = QDTWorkingMemory(
        QDTWorkingMemoryConfig(
            input_dim=32,
            hidden_dim=64,
            num_depths=8,
            num_slots=8,
            num_heads=4,
            enable_chart_fusion_policy=True,
            chart_fusion_condition_mix=0.0,
        )
    )
    x = torch.randn(2, 5, 32)
    y = on(x, operation="read", context_map_name="hierarchical")
    assert torch.isfinite(y).all()
    assert on.get_metrics()["pfh_enabled"] == 1.0
    assert on.get_metrics()["cfp_enabled"] == 1.0
    assert on.dual_fusion.ltm.last_output.handoff.space == FUSION_SPACE
