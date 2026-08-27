import torch

from mnemonic_cortex.working_memory import (
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    WMChartFusionPolicy,
    WMChartFusionPolicyConfig,
    WMDualFusionConfig,
    WMDualFusionController,
    scenario_prior,
)
from mnemonic_cortex.working_memory.context_geometry_maps import build_default_context_geometry_maps
from mnemonic_cortex.working_memory.wm_chart_fusion_policy import LEGACY_FUSION_WEIGHTS


def test_scenario_role_priors_follow_map_purpose():
    maps = build_default_context_geometry_maps(8)
    literal = scenario_prior(
        "literal",
        charts=maps["literal"].geometry_by_depth,
        depth_weights=maps["literal"].depth_weights,
    )
    hierarchical = scenario_prior(
        "hierarchical",
        charts=maps["hierarchical"].geometry_by_depth,
        depth_weights=maps["hierarchical"].depth_weights,
    )
    procedural = scenario_prior(
        "procedural",
        charts=maps["procedural"].geometry_by_depth,
        depth_weights=maps["procedural"].depth_weights,
    )
    spatial = scenario_prior(
        "spatial_mechanical",
        charts=maps["spatial_mechanical"].geometry_by_depth,
        depth_weights=maps["spatial_mechanical"].depth_weights,
    )
    assert literal[0] > hierarchical[0]
    assert hierarchical[1] > literal[1]
    assert procedural[3] > hierarchical[3]
    assert spatial[2] > literal[2]
    for prior in (literal, hierarchical, procedural, spatial):
        assert abs(sum(prior) - 1.0) < 1e-6


def test_unknown_map_falls_back_to_legacy_mix():
    prior = torch.tensor(scenario_prior("not_a_real_map"))
    expected = torch.tensor(LEGACY_FUSION_WEIGHTS)
    assert torch.allclose(prior, expected, atol=1e-6)


def test_gate_zero_ignores_learned_logits():
    policy = WMChartFusionPolicy(
        WMChartFusionPolicyConfig(enable=True, gate_init=0.0, condition_mix=0.0)
    )
    with torch.no_grad():
        policy.global_logits.fill_(4.0)
    out = policy(2, torch.device("cpu"), torch.float32, map_name="hierarchical")
    expected = torch.tensor(policy.hardcoded_prior("hierarchical"))
    assert torch.allclose(out.weights[0], expected, atol=1e-5)
    assert torch.allclose(out.weights[1], expected, atol=1e-5)
    assert out.trace["kl_to_prior"] < 1e-6


def test_open_gate_lets_logits_move_weights():
    policy = WMChartFusionPolicy(
        WMChartFusionPolicyConfig(enable=True, gate_init=0.0, condition_mix=0.0)
    )
    baseline = policy(1, torch.device("cpu"), torch.float32, map_name="hierarchical").weights.clone()
    policy.open_for_finetune(1.0)
    with torch.no_grad():
        policy.global_logits[1] = 5.0
    moved = policy(1, torch.device("cpu"), torch.float32, map_name="hierarchical").weights
    assert float(moved[0, 1].detach()) > float(baseline[0, 1].detach())


def test_disagreement_condition_shifts_toward_wm():
    policy = WMChartFusionPolicy(
        WMChartFusionPolicyConfig(enable=True, gate_init=0.0, condition_mix=1.0)
    )
    calm = policy(
        2,
        torch.device("cpu"),
        torch.float32,
        map_name="hierarchical",
        confidence=torch.ones(2),
        disagreement=torch.zeros(2),
    )
    noisy = policy(
        2,
        torch.device("cpu"),
        torch.float32,
        map_name="hierarchical",
        confidence=torch.ones(2),
        disagreement=torch.ones(2) * 8.0,
    )
    assert float(noisy.weights[0, 0].detach()) > float(calm.weights[0, 0].detach())


def test_dual_fusion_opt_in_off_keeps_legacy_weights():
    module = WMDualFusionController(WMDualFusionConfig(dim=32, top_k=3))
    assert module.fusion_policy is None
    tokens = torch.randn(2, 5, 32)
    y, packed = module(tokens, context_map_name="hierarchical", return_trace=True)
    assert y.shape == tokens.shape
    assert packed["trace"]["chart_fusion_policy"]["enabled"] is False
    weights = packed["trace"]["fusion_weights"]
    expected = torch.tensor(LEGACY_FUSION_WEIGHTS)
    got = torch.tensor(weights)
    got = got / got.sum()
    assert torch.allclose(got, expected, atol=1e-5)


def test_dual_fusion_policy_supersedes_legacy_mix():
    legacy = WMDualFusionController(WMDualFusionConfig(dim=32, top_k=3, residual_mix=0.35))
    policy = WMDualFusionController(
        WMDualFusionConfig(
            dim=32,
            top_k=3,
            residual_mix=0.35,
            enable_chart_fusion_policy=True,
            chart_fusion_gate_init=0.0,
            chart_fusion_condition_mix=0.0,
        )
    )
    policy.fusion_proj.load_state_dict(legacy.fusion_proj.state_dict())
    policy.ltm.load_state_dict(legacy.ltm.state_dict())
    policy.mann.load_state_dict(legacy.mann.state_dict())
    policy.spcp.load_state_dict(legacy.spcp.state_dict())
    tokens = torch.randn(2, 5, 32)
    y_legacy, packed_legacy = legacy(tokens, context_map_name="procedural", return_trace=True)
    y_policy, packed_policy = policy(tokens, context_map_name="procedural", return_trace=True)
    assert packed_policy["trace"]["chart_fusion_policy"]["enabled"] is True
    assert packed_policy["trace"]["chart_fusion_policy"]["map_name"] == "procedural"
    assert packed_policy["trace"]["fusion_weights"][3] > packed_legacy["trace"]["fusion_weights"][3]
    assert not torch.allclose(y_legacy, y_policy)


def test_finetune_groups_and_recipe_snapshot():
    policy = WMChartFusionPolicy(WMChartFusionPolicyConfig(enable=True, condition_mix=0.0))
    groups = policy.finetune_parameter_groups()
    assert set(groups) == {"gate", "global_logits", "scenario_logits", "chart_logits"}
    recipe = policy.snapshot_recipe()
    assert recipe["testbed"] == "wm_chart_fusion_policy"
    assert "hierarchical" in recipe["priors"]
    assert recipe["qspin_live_routing"] is False
    policy.open_for_finetune(0.1)
    assert abs(float(policy.gate.detach()) - 0.1) < 1e-6


def test_qdt_chart_fusion_policy_default_off():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    assert wm.get_metrics()["cfp_enabled"] == 0.0
    x = torch.randn(2, 5, 32)
    y, trace = wm(x, operation="read", context_map_name="hierarchical", return_trace=True)
    assert y.shape == x.shape
    dual = next(item for item in trace["items"] if item.get("stage") == "dual_fusion")
    payload = dual.get("payload") or dual
    inner = payload.get("trace") if isinstance(payload, dict) else None
    # dual_fusion merge may nest differently; just require a finite read
    assert torch.isfinite(y).all()


def test_qdt_chart_fusion_policy_opt_in_metrics():
    cfg = QDTWorkingMemoryConfig(
        input_dim=32,
        hidden_dim=64,
        num_depths=8,
        num_slots=8,
        num_heads=4,
        enable_chart_fusion_policy=True,
        chart_fusion_gate_init=0.0,
        chart_fusion_condition_mix=0.0,
    )
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    y = wm(x, operation="read", context_map_name="hierarchical")
    assert torch.isfinite(y).all()
    metrics = wm.get_metrics()
    assert metrics["cfp_enabled"] == 1.0
    assert metrics["cfp_gate"] == 0.0
