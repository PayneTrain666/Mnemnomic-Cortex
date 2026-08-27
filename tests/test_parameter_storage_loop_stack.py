import torch
import tempfile
from pathlib import Path

from mnemonic_cortex import (
    DEFAULT_PARAMETER_LOOP_MANIFOLDS,
    ParameterStorageLoopConfig,
    ParameterStorageLoopStack,
    estimate_parameter_storage_loop_capacity,
)
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.trainable_parameter_cps import (
    TrainableParameterCPS,
    TrainableParameterCPSConfig,
)


def test_parameter_storage_loop_stack_forward_trace_and_shape():
    cfg = ParameterStorageLoopConfig(
        model_dim=32,
        parameter_slots_per_layer=6,
        free_hidden_layers=2,
        num_heads=4,
    )
    model = ParameterStorageLoopStack(cfg)
    x = torch.randn(2, 5, 32)
    out, trace = model(x, return_trace=True)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert trace["trace_type"] == "parameter_storage_loop_stack"
    assert len(trace["visible_layers"]) == 10
    assert len(trace["hidden_storage_layers"]) == 10
    assert trace["visible_layers"][0]["manifold"] == "hyperbolic"
    assert trace["visible_layers"][1]["manifold"] == "spatial_s3"
    assert trace["visible_layers"][3]["manifold"] == "complex_projective_kahler"
    assert trace["visible_layers"][-1]["manifold"] == "quaternion_spatial_loop"
    assert trace["loop_attention_tokens"] == 31
    assert trace["safety"]["shared_slot_write"] is False


def test_parameter_storage_loop_estimate_uses_product_manifold_accounting():
    cfg = ParameterStorageLoopConfig(model_dim=16, parameter_slots_per_layer=4, free_hidden_layers=0)
    estimate = estimate_parameter_storage_loop_capacity(cfg)

    normal = estimate["normal_physical_slot_scalars"]
    effective = estimate["effective_parameter_storage_units"]
    assert normal == 20 * 4 * 16
    assert effective > normal
    assert estimate["effective_to_physical_ratio"] > 1.0
    assert estimate["product_manifold_effective_units"] > estimate["pair_loop_effective_units"]
    assert estimate["accounting"] == "loop_stack_product_manifold_effective_storage"
    assert estimate["manifold_stack"] == list(DEFAULT_PARAMETER_LOOP_MANIFOLDS)


def test_parameter_storage_loop_rejects_non_ten_layer_maps():
    cfg = ParameterStorageLoopConfig(manifold_stack=("hyperbolic",) * 9)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "exactly 10" in str(exc)
    else:
        raise AssertionError("expected non-10 manifold map to be rejected")


def test_cortex_optional_parameter_storage_loop_wires_hidden_attention_source():
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        enable_parameter_storage_loop_stack=True,
        parameter_loop_slots_per_layer=2,
        parameter_loop_free_hidden_layers=1,
    )
    model.eval()

    assert model.parameter_storage_loop_stack is not None
    desc = model.describe_parameter_storage_loop()
    assert desc["enabled"] is True
    assert desc["hidden_attention_source"] == "cortex.parameter_loop"
    assert desc["capacity_estimate"]["effective_parameter_storage_units"] > desc["capacity_estimate"]["normal_physical_slot_scalars"]
    assert any(name.startswith("cortex.parameter_loop") for name, _module in model.global_hidden_orchestrator._tracked_modules)

    x = torch.randn(1, 2, 16)
    ctx = torch.randn(1, 16)
    with torch.no_grad():
        out = model(x, ctx, operation="process")
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert model.last_parameter_storage_loop_stats["effective_to_physical_ratio"] > 1.0
    assert model.last_parameter_storage_loop_stats["ltm_context_tokens"] > 0
    assert model.long_term_memory.external_attention_context is not None


def test_parameter_storage_loop_lightbulb_and_guarded_training_updates():
    cfg = ParameterStorageLoopConfig(
        model_dim=16,
        parameter_slots_per_layer=2,
        free_hidden_layers=0,
        enable_training_slot_updates=True,
        training_update_lr=0.05,
        lightbulb_threshold=0.0,
    )
    model = ParameterStorageLoopStack(cfg)
    model.train()
    before = model.visible_parameter_slots.detach().clone()
    x = torch.randn(2, 3, 16)
    out, trace = model(
        x,
        fire_mask=torch.tensor([True, False]),
        recall_boost=0.5,
        allow_slot_update=True,
        slot_update_scale=1.0,
        return_trace=True,
    )

    assert out.shape == x.shape
    assert trace["lightbulb"]["triggered"] is True
    assert trace["slot_update"]["updated"] is True
    assert not torch.equal(before, model.visible_parameter_slots.detach())


def test_parameter_storage_loop_training_updates_default_to_disabled():
    cfg = ParameterStorageLoopConfig(model_dim=16, parameter_slots_per_layer=2)
    model = ParameterStorageLoopStack(cfg)
    model.train()
    before = model.visible_parameter_slots.detach().clone()
    _out, trace = model(torch.randn(1, 2, 16), allow_slot_update=True, return_trace=True)

    assert trace["slot_update"]["updated"] is False
    assert torch.equal(before, model.visible_parameter_slots.detach())


def test_parameter_storage_loop_consolidates_referenced_shadow_bundles():
    cfg = ParameterStorageLoopConfig(model_dim=16, parameter_slots_per_layer=2, training_update_lr=0.1)
    store = ParameterStorageLoopStack(cfg)
    source = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16))
    before = store.visible_parameter_slots.detach().clone()
    source_param_ids = {name: id(param) for name, param in source.named_parameters()}

    trace = store.consolidate_parameter_bundles(
        source.named_parameters(),
        trigger_state={"step": 7, "factors": ("unit_test", "shape_rank")},
        max_bundles=2,
        min_total_numel=1,
        write_scale=1.0,
    )

    assert trace["mode"] == "referenced_shadow"
    assert trace["created_bundle_count"] > 0
    assert trace["optimizer_safe"] is True
    assert trace["original_parameters_replaced"] is False
    assert not torch.equal(before, store.visible_parameter_slots.detach())
    state = store.parameter_bundle_registry_state()
    assert state["registry_size"] == trace["created_bundle_count"]
    refs = next(iter(state["bundles"].values()))["refs"]
    assert refs
    assert {name: id(param) for name, param in source.named_parameters()} == source_param_ids


def test_parameter_loop_uses_stable_cps_handles_without_reconsolidating_slabs():
    source = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    cps = TrainableParameterCPS(
        TrainableParameterCPSConfig(include=("*",), enable_compression=False)
    )
    cps.commit(cps.stage(source))
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
    optimizer_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    slab_values = [param.detach().clone() for param in cps.parameters()]
    store = ParameterStorageLoopStack(
        ParameterStorageLoopConfig(model_dim=16, parameter_slots_per_layer=2)
    )

    trace = store.consolidate_parameter_bundles(
        source.named_parameters(),
        cps_registry=cps.registry,
        cps_resolver=cps.materialize,
        max_bundles=4,
    )

    refs = [
        ref
        for bundle in trace["created_bundles"]
        for ref in bundle["refs"]
    ]
    assert refs
    assert all(ref["cps_handle"] in cps.registry for ref in refs)
    assert all("_trainable_parameter_cps" not in ref["name"] for ref in refs)
    assert trace["accounting"]["literal_named_parameter_numel"] == 0
    assert trace["accounting"]["referenced_cps_numel"] > 0
    assert trace["accounting"]["cps_weights_copied"] is False
    assert {
        id(param) for group in optimizer.param_groups for param in group["params"]
    } == optimizer_ids
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(slab_values, cps.parameters())
    )


def test_cortex_auto_parameter_consolidation_trigger_preserves_parameter_ids():
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        enable_parameter_storage_loop_stack=True,
        parameter_loop_slots_per_layer=2,
        parameter_loop_free_hidden_layers=0,
        enable_parameter_loop_auto_consolidation=True,
        parameter_loop_consolidation_interval=1,
        parameter_loop_consolidation_max_bundles=2,
        parameter_loop_consolidation_min_total_numel=1,
    )
    model.train()
    before_ids = {name: id(param) for name, param in model.named_parameters()}

    trace = model._maybe_consolidate_trainable_parameters()

    assert trace["triggered"] is True
    assert trace["created_bundle_count"] > 0
    assert model.describe_parameter_storage_loop()["bundle_registry"]["registry_size"] > 0
    after_ids = {name: id(param) for name, param in model.named_parameters()}
    assert before_ids == after_ids


def test_parameter_loop_gate_regularization_keeps_gate_in_safe_band():
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        enable_parameter_storage_loop_stack=True,
        parameter_loop_slots_per_layer=2,
        parameter_loop_free_hidden_layers=1,
        parameter_loop_gate_min=0.08,
        parameter_loop_gate_max=0.35,
    )

    with torch.no_grad():
        model.parameter_storage_loop_gate.fill_(-20.0)
    assert model.parameter_loop_gate_regularization().item() > 0.0

    with torch.no_grad():
        model.parameter_storage_loop_gate.fill_(20.0)
    assert model.parameter_loop_gate_regularization().item() > 0.0

    with torch.no_grad():
        model.parameter_storage_loop_gate.fill_(
            torch.logit(torch.tensor(0.20)).item()
        )
    assert model.parameter_loop_gate_regularization().item() == 0.0


def test_parameter_loop_zero_gate_is_identity_residual():
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        enable_parameter_storage_loop_stack=True,
        parameter_loop_slots_per_layer=2,
        parameter_loop_free_hidden_layers=1,
    )
    model.eval()
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        model.parameter_storage_loop_gate.fill_(-30.0)
        out = model._apply_parameter_storage_loop(x, phase="identity_test")

    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_parameter_loop_bundle_registry_survives_checkpoint_roundtrip():
    def build_model():
        return EnhancedMnemonicCortex(
            input_dim=16,
            output_dim=16,
            ltm_hg_slots=16,
            ltm_cgmn_slots=16,
            ltm_curved_slots=8,
            ltm_spatial_slots=8,
            enable_parameter_storage_loop_stack=True,
            parameter_loop_slots_per_layer=2,
            parameter_loop_free_hidden_layers=0,
            enable_parameter_loop_auto_consolidation=True,
            parameter_loop_consolidation_interval=1,
            parameter_loop_consolidation_max_bundles=1,
            parameter_loop_consolidation_min_total_numel=1,
        )

    model = build_model()
    model.train()
    trace = model._maybe_consolidate_trainable_parameters()
    assert trace["created_bundle_count"] == 1
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "param_loop.pt"
        model.save_checkpoint(str(path))
        loaded = build_model()
        loaded.load_checkpoint(str(path), strict=True)

    registry = loaded.describe_parameter_storage_loop()["bundle_registry"]
    assert registry["registry_size"] == 1
    assert next(iter(registry["bundles"].values()))["refs"]


def test_cortex_cps_handle_bundle_registry_survives_checkpoint_roundtrip():
    def build_model():
        return EnhancedMnemonicCortex(
            input_dim=16,
            output_dim=16,
            ltm_hg_slots=16,
            ltm_cgmn_slots=16,
            ltm_curved_slots=8,
            ltm_spatial_slots=8,
            enable_parameter_storage_loop_stack=True,
            parameter_loop_slots_per_layer=2,
            parameter_loop_free_hidden_layers=0,
            enable_parameter_loop_auto_consolidation=True,
            parameter_loop_consolidation_interval=1,
            parameter_loop_consolidation_max_bundles=2,
            parameter_loop_consolidation_min_total_numel=1,
            parameter_loop_consolidation_include=("ctx_proj",),
        )

    model = build_model()
    cps = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(include=("ctx_proj",))
    )
    cps.commit(cps.stage(model))
    model.train()
    trace = model._maybe_consolidate_trainable_parameters()
    refs = [
        ref
        for bundle in trace["created_bundles"]
        for ref in bundle["refs"]
    ]
    assert refs and all(ref["cps_handle"] for ref in refs)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cps_parameter_loop.pt"
        model.save_checkpoint(str(path))
        loaded = build_model()
        loaded.load_checkpoint(str(path), strict=True)

    loaded_refs = [
        ref
        for bundle in loaded.describe_parameter_storage_loop()["bundle_registry"][
            "bundles"
        ].values()
        for ref in bundle["refs"]
    ]
    assert sorted(ref["cps_handle"] for ref in loaded_refs) == sorted(
        ref["cps_handle"] for ref in refs
    )
