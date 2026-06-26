import torch

from mnemonic_cortex.consolidated_memory import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from mnemonic_cortex.consolidated_memory_depth_stack import (
    CMS_SUPER_PRODUCT_MANIFOLD,
    ConsolidatedMemoryDepthCfg,
    ConsolidatedMemoryDepthStack,
    DEFAULT_CMS_DEPTH_MANIFOLDS,
    DEFAULT_CMS_VISIBLE_MANIFOLD_STACK,
    estimate_consolidated_memory_depth_capacity,
)
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_cms_depth_stack_visible_manifold_order():
    cfg = ConsolidatedMemoryDepthCfg(
        model_dim=32,
        memory_slots_per_layer=4,
        free_hidden_layers=2,
        num_heads=4,
    )
    assert list(cfg.depth_manifold_stack) == list(DEFAULT_CMS_DEPTH_MANIFOLDS)
    assert cfg.depth_manifold_stack[0] == "euclidean"
    assert cfg.depth_manifold_stack[1] == "hyperbolic"
    assert cfg.depth_manifold_stack[2] == "spatial_s3"
    assert cfg.depth_manifold_stack[3] == "complex_projective"
    assert cfg.depth_manifold_stack[4] == "complex_projective_kahler"
    assert cfg.depth_manifold_stack[5] == "toroidal"
    assert cfg.depth_manifold_stack[6] == "grassmann_subspace"
    assert cfg.depth_manifold_stack[7] == "quaternion_spatial_loop"
    assert cfg.visible_manifold_stack[-1] == CMS_SUPER_PRODUCT_MANIFOLD


def test_cms_depth_stack_forward_trace_and_shape():
    cfg = ConsolidatedMemoryDepthCfg(model_dim=32, memory_slots_per_layer=4, free_hidden_layers=2, num_heads=4)
    model = ConsolidatedMemoryDepthStack(cfg)
    x = torch.randn(2, 5, 32)
    out, trace = model(x, return_trace=True)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert trace["trace_type"] == "consolidated_memory_depth_stack"
    assert len(trace["visible_layers"]) == 9
    assert len(trace["hidden_storage_layers"]) == 9
    assert trace["visible_layers"][0]["manifold"] == "euclidean"
    assert trace["visible_layers"][7]["manifold"] == "quaternion_spatial_loop"
    assert trace["visible_layers"][8]["manifold"] == CMS_SUPER_PRODUCT_MANIFOLD
    assert trace["qh_num_depths"] == 8


def test_cms_depth_capacity_uses_super_product_accounting():
    cfg = ConsolidatedMemoryDepthCfg(model_dim=16, memory_slots_per_layer=4, free_hidden_layers=0)
    estimate = estimate_consolidated_memory_depth_capacity(cfg)

    normal = estimate["normal_physical_slot_scalars"]
    effective = estimate["effective_memory_storage_units"]
    assert normal == 18 * 4 * 16
    assert effective > normal
    assert estimate["super_product_effective_units"] > 0.0
    assert estimate["depth_product_effective_units"] > 0.0
    assert estimate["accounting"] == "cms_depth_loop_stack_super_product_manifold"


def test_consolidated_memory_store_depth_stack_write_and_read():
    store = ConsolidatedMemoryStore(
        ConsolidatedMemoryCfg(d_model=32, enable_depth_stack=True, memory_slots_per_layer=4, depth_free_hidden_layers=1)
    )
    desc = store.describe_depth_stack()
    assert desc["enabled"] is True
    assert desc["depth_manifold_stack"][0] == "euclidean"

    candidate = {
        "E": torch.randn(32),
        "H": torch.randn(64),
        "P": (torch.ones(32) * 0.5, torch.zeros(32)),
    }
    store.write("k1", candidate, alpha=0.5)
    fused = store.read("k1")
    assert fused.shape == (32,)
    assert torch.isfinite(fused).all()


def test_cortex_advanced_cms_depth_stack_wires_hidden_orchestrator():
    model = EnhancedMnemonicCortex(
        input_dim=32,
        output_dim=32,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        global_hidden_max_layers=64,
    )
    model.enable_advanced_consolidation()

    assert model.advanced_cms is not None
    assert model.advanced_cms.depth_stack is not None
    desc = model.describe_consolidated_memory_depth_stack()
    assert desc["enabled"] is True
    assert desc["hidden_attention_source"] == "cortex.cms_depth_stack"
    assert desc["capacity_estimate"]["effective_memory_storage_units"] > desc["capacity_estimate"]["normal_physical_slot_scalars"]

    x = torch.randn(2, 4, 32)
    model.eval()
    with torch.no_grad():
        out = model._apply_consolidated_memory_depth(x, phase="test")
    assert out.shape == x.shape
    assert model.last_cms_depth_stack_stats.get("enabled") is True

    metrics = model.get_metrics()
    assert metrics["cms_depth_stack_enabled"] == 1.0
