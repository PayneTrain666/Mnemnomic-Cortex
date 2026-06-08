import torch

import mnemonic_cortex.ltm as ltm
import mnemonic_cortex.ltm.cgmn_semantic_ltm as cgmn_semantic_ltm
import mnemonic_cortex.ltm.spatial_ltm_mann as spatial_ltm_mann
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.ltm import (
    DualTransformerPolicy,
    EnhancedSpatialMnemonicCortex,
    ReasoningStack,
    TransformerStack,
    default_config,
    metric_distance,
)


def test_ltm_package_exports_all_modules():
    names = [
        "EpisodeRecord",
        "HGEpisodicLTM",
        "DualTransformerPolicy",
        "SpatialLtmMannConfig",
        "default_config",
        "SharedValueStore",
        "DepthRouter",
        "GeometryMemoryBank",
        "LTMSubsystem",
        "TripleHybridLTM",
        "MANNReasoner",
        "WorkingMemory",
        "EnhancedSpatialMnemonicCortex",
        "TransformerStack",
        "ReasoningStack",
        "metric_distance",
        "log_map",
        "frechet_mean",
    ]
    for name in names:
        assert hasattr(ltm, name), f"missing export: {name}"


def test_spatial_ltm_mann_alias_package_reexports_parent_api():
    assert spatial_ltm_mann.EnhancedSpatialMnemonicCortex is ltm.EnhancedSpatialMnemonicCortex
    assert spatial_ltm_mann.DualTransformerPolicy is ltm.DualTransformerPolicy
    assert spatial_ltm_mann.ReasoningStack is ltm.ReasoningStack


def test_cgmn_semantic_ltm_alias_package_reexports_parent_api():
    assert cgmn_semantic_ltm.TripleHybridLTM is ltm.TripleHybridLTM
    assert cgmn_semantic_ltm.DEFAULT_CGMN_DEPTH_CHART is ltm.DEFAULT_CGMN_DEPTH_CHART
    assert cgmn_semantic_ltm.DualTransformerPolicy is ltm.DualTransformerPolicy


def test_enhanced_spatial_mnemonic_cortex_dual_stack_and_reasoning():
    cfg = default_config(
        input_dim=32,
        model_dim=32,
        output_dim=32,
        value_dim=32,
        key_dim=16,
        shared_slots=32,
        bank_transformer_layers=0,
        fixed_transformer_layers=2,
        inherited_bank_layers=3,
        wm_tf_depth=0,
        reasoning_stack_depth=0,
        wm_slots=4,
        wm_dim=32,
    )
    model = EnhancedSpatialMnemonicCortex(cfg)
    assert model.transformer_policy.bank_layers == 3
    assert model.transformer_policy.fixed_layers == 2
    assert model.wm.wm_tf_depth == 3
    assert model.reasoning_stack is not None

    x = torch.randn(2, 32)
    out, traces = model(x, operation="process", return_traces=True)
    assert out.shape == (2, 32)
    assert traces["transformer_policy"]["dual_stack_active"] is True
    assert traces["reasoning_stack_depth"] == 2


def test_manifold_utils_metric_distance_runs():
    q = torch.randn(2, 16)
    k = torch.randn(2, 8, 16)
    for geom in ("euclid", "hyper", "sphere", "torus", "spatial"):
        d = metric_distance(geom, q, k)
        assert d.shape == (2, 8)
        assert torch.isfinite(d).all()


def test_cortex_spatial_extension_inherits_ltm_policy():
    cortex = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_enable_spatial_ltm=True,
        ltm_spatial_transformer_layers=0,
        ltm_spatial_fixed_transformer_layers=3,
        ltm_n_transformer_layers=3,
    )
    ext = cortex.spatial_ltm_extension
    assert ext is not None
    assert ext.transformer_policy.bank_layers == 3
    assert ext.transformer_policy.fixed_layers == 3
    assert ext.wm.wm_tf_depth == 3
    assert ext.reasoning_stack is not None

    x = torch.randn(1, 16)
    out, traces = cortex.run_spatial_ltm_extension(x, operation="process", return_traces=True)
    assert out.shape == (1, 16)
    assert traces["transformer_policy"]["bank_layers"] == 3
    assert traces["transformer_policy"]["fixed_layers"] == 3
    assert traces["reasoning_stack_depth"] == 3
