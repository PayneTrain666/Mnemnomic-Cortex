import torch

from mnemonic_cortex.utils import fast_pairwise_l2, seed_everything
from mnemonic_cortex.working_memory.context_geometry_maps import (
    ContextGeometryMap,
    build_default_context_geometry_maps,
    validate_context_geometry_map,
)
from mnemonic_cortex.working_memory.wm_context_mount import GeometryMountedContextBuffer


def test_fast_pairwise_l2_rejects_dim_mismatch():
    a = torch.randn(3, 5)
    b = torch.randn(7, 6)
    try:
        _ = fast_pairwise_l2(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched feature dimensions")


def test_seed_everything_reproducible_cpu():
    seed_everything(1234, deterministic=True)
    x1 = torch.randn(2, 3)
    seed_everything(1234, deterministic=True)
    x2 = torch.randn(2, 3)
    assert torch.allclose(x1, x2)


def test_context_geometry_map_normalization_and_validation():
    custom = ContextGeometryMap(
        name="custom",
        purpose="unit test",
        geometry_by_depth=["euclidean", "spherical"],
        depth_weights=[2.0, 0.0],
        warmup_weights=[0.0],
        trainable_weight_init=[0.0],
        mount_strategy="phase_bias",
        triplet_bias=[1.0, 0.5, 0.25],
    ).normalized_for_depths(8)
    validate_context_geometry_map(custom, num_depths=8)
    assert len(custom.depth_weights) == 8
    assert abs(sum(custom.depth_weights) - 1.0) < 1e-6
    assert len(custom.warmup_weights) == 8
    assert abs(sum(custom.warmup_weights) - 1.0) < 1e-6


def test_context_mount_trace_is_not_shared_between_calls():
    maps = build_default_context_geometry_maps(num_depths=8)
    buf = GeometryMountedContextBuffer(dim=16, num_depths=8, maps=maps)
    ctx = torch.randn(2, 3, 16)
    depth = torch.randn(2, 8, 2, 3, 16)
    _, selected1 = buf.mount(ctx, depth, requested_map="literal")
    _, selected2 = buf.mount(ctx, depth, requested_map="literal")
    assert hasattr(selected1, "trace")
    assert hasattr(selected2, "trace")
    assert selected1 is not selected2
    assert selected1.trace is not selected2.trace
