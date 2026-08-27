import torch

from geometry.chart_native import (
    blender_priors_from_charts,
    majority_chart,
    project_with_residual,
    resolve_chart_geom,
)
from geometry.manifold_utils import geometry_name_to_geom
from mnemonic_cortex.reasoning_depth import (
    LTMDepthAdapter,
    LTMDepthAdapterConfig,
    MANNDepthAdapter,
    MANNDepthAdapterConfig,
    MANNLTMSharedSlotGeometry,
    SharedGeometrySlotConfig,
)
from mnemonic_cortex.working_memory import (
    CurvedSlotStateBank,
    CurvedSlotStateConfig,
    CurvatureMetricPolicy,
    CurvatureMetricPolicyConfig,
    DepthSpecificAddressing,
    DepthSpecificAddressingConfig,
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
)
from mnemonic_cortex.working_memory.wm_native_chart_geometry import project_depth_replicas


def test_euclidean_projection_is_identity_at_full_mix():
    x = torch.randn(4, 16)
    y, meta = project_with_residual(x, "euclidean", mix=1.0)
    assert meta["geom"] == "euclid"
    assert torch.allclose(y, x)


def test_chart_projection_mix_zero_is_identity():
    x = torch.randn(4, 16)
    for name in ("hyperbolic", "spherical", "torus", "quaternion", "holographic_phase"):
        y, _ = project_with_residual(x, name, mix=0.0)
        assert torch.allclose(y, x)


def test_grassmannian_alias_and_vector_safe_sphere():
    assert geometry_name_to_geom("grassmannian") == "grassmann"
    x = torch.randn(3, 16)
    assert resolve_chart_geom("grassmannian", x) == "sphere"
    assert resolve_chart_geom("complex_projective", x) == "sphere"
    y, meta = project_with_residual(x, "subspace", mix=1.0)
    assert meta["geom"] == "sphere"
    assert torch.isfinite(y).all()
    assert y.shape == x.shape


def test_blender_priors_follow_chart_histogram():
    priors = blender_priors_from_charts(["hyperbolic", "hyperbolic", "euclidean", "torus"])
    assert priors.shape == (6,)
    assert torch.allclose(priors.sum(), torch.tensor(1.0))
    assert float(priors[0]) == 0.5
    assert majority_chart(["euclidean", "hyperbolic", "hyperbolic"]) == "hyperbolic"


def test_project_depth_replicas_mix_zero_identity():
    depth_state = torch.randn(2, 8, 4, 3, 16)
    charts = ["euclidean", "hyperbolic", "spherical", "torus", "complex", "subspace", "spatial_se3", "spcp"]
    out, trace = project_depth_replicas(depth_state, charts, mix=0.0)
    assert torch.allclose(out, depth_state)
    assert trace["enabled"] is False


def test_project_depth_replicas_is_finite_and_shaped():
    depth_state = torch.randn(2, 8, 4, 3, 16)
    charts = ["euclidean", "hyperbolic", "spherical", "torus", "complex", "subspace", "spatial_se3", "spcp"]
    out, trace = project_depth_replicas(depth_state, charts, mix=0.2)
    assert out.shape == depth_state.shape
    assert torch.isfinite(out).all()
    assert trace["enabled"] is True
    assert not torch.allclose(out, depth_state)


def test_depth_addressing_native_chart_keeps_activation_contract():
    dim, slots, depths = 32, 6, 8
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=slots, dim=dim))
    policy = CurvatureMetricPolicy(CurvatureMetricPolicyConfig(num_slots=slots, num_depths=depths, context_dim=dim))
    addressing = DepthSpecificAddressing(
        DepthSpecificAddressingConfig(
            dim=dim,
            num_slots=slots,
            num_depths=depths,
            top_k=3,
            native_chart_residual_mix=0.2,
        ),
        slot_bank=bank,
        curvature_policy=policy,
    )
    depth_state = torch.randn(2, 8, 5, 3, 32)
    out, trace = addressing(depth_state, context_map_name="hierarchical", return_trace=True)
    assert out.activation.shape == (2, 8, 6)
    assert torch.isfinite(out.activation).all()
    assert torch.allclose(out.activation.sum(dim=-1), torch.ones(2, 8), atol=1e-5)
    assert trace["paamax_metadata"]["native_chart_residual_mix"] == 0.2


def test_qdt_native_chart_geometry_read_is_finite():
    cfg = QDTWorkingMemoryConfig(
        input_dim=32,
        hidden_dim=64,
        num_depths=8,
        num_slots=8,
        num_heads=4,
        enable_native_chart_geometry=True,
        native_chart_residual_mix=0.2,
    )
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    y, trace = wm(x, operation="read", context_map_name="hierarchical", return_trace=True)
    stages = {item["stage"] for item in trace["items"]}
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert "native_chart_geometry" in stages
    metrics = wm.get_metrics()
    assert metrics["ncg_enabled"] == 1.0
    assert metrics["ncg_mix"] == 0.2


def test_qdt_native_chart_mix_zero_skips_projection_stage():
    cfg = QDTWorkingMemoryConfig(
        input_dim=32,
        hidden_dim=64,
        num_depths=8,
        num_slots=8,
        num_heads=4,
        enable_native_chart_geometry=True,
        native_chart_residual_mix=0.0,
        enable_inter_manifold_attention=False,
    )
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    y, trace = wm(x, operation="read", context_map_name="literal", return_trace=True)
    stages = {item["stage"] for item in trace["items"]}
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert "native_chart_geometry" not in stages
    assert wm.get_metrics()["ncg_enabled"] == 0.0


def test_ltm_bank_chart_distance_mix_zero_keeps_base():
    from geometry.chart_native import configure_memory_bank_charts, mix_bank_chart_distance
    from mnemonic_cortex.memory_curved import EnhancedCurvedMemory

    bank = EnhancedCurvedMemory(input_dim=16, hidden_dim=16, mem_slots=8, transformer_layers=0)
    configure_memory_bank_charts(bank, ["hyperbolic"] * 8, mix=0.0)
    query = torch.randn(2, 16)
    keys = torch.randn(2, 3, 16)
    base = torch.rand(2, 3) + 0.1
    mixed = mix_bank_chart_distance(bank, query, keys, base)
    assert torch.allclose(mixed, base)


def test_mann_shared_slot_native_chart_mix_zero_is_identity():
    dim = 16
    shared = MANNLTMSharedSlotGeometry(
        config=SharedGeometrySlotConfig(
            enabled=True,
            key_dim=dim,
            value_dim=dim,
            native_chart_residual_mix=0.0,
        ),
        mann_adapter=MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12)),
        ltm_adapter=LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12)),
    )
    tensor = torch.randn(2, 16)
    out, meta = shared._chart_transform(tensor, geometry_map_name="procedural", depth_index=3)
    assert torch.allclose(out, tensor)
    assert meta["native_chart"] is True
    assert meta["residual_mix"] == 0.0


def test_mann_shared_slot_native_chart_is_finite():
    dim = 16
    shared = MANNLTMSharedSlotGeometry(
        config=SharedGeometrySlotConfig.enabled_default(key_dim=dim, value_dim=dim),
        mann_adapter=MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12)),
        ltm_adapter=LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12)),
    )
    query = torch.randn(2, 4, 16)
    out, trace = shared.run_shared_reasoning(
        query,
        content="native chart hop",
        mann_slot_index=2,
        ltm_slot_index=5,
        hop_id=1,
        return_trace=True,
    )
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    assert trace["mann_chart_transform"]["native_chart"] is True
