import torch

from mnemonic_cortex import (
    DEFAULT_PARAMETER_LOOP_MANIFOLDS,
    ParameterStorageLoopConfig,
    ParameterStorageLoopStack,
    estimate_parameter_storage_loop_capacity,
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
