import torch

from mnemonic_cortex.working_memory.context_geometry_maps import (
    build_default_context_geometry_maps,
    validate_context_geometry_map,
)


def test_all_required_context_maps_exist_and_validate():
    maps = build_default_context_geometry_maps(num_depths=8)
    required = {
        "literal",
        "hierarchical",
        "temporal",
        "spatial_mechanical",
        "symbolic_mathematical",
        "procedural",
        "conflict_verification",
        "creative_synthesis",
        "policy_governance",
        "quantum_holographic",
    }
    assert required.issubset(set(maps))
    for spec in maps.values():
        validate_context_geometry_map(spec, num_depths=8)
        assert len(spec.depth_weights) == 8
        assert len(spec.warmup_weights) == 8
        assert len(spec.trainable_weight_init) == 8
        assert len(spec.triplet_bias) == 3


def test_full_geometry_catalogue_is_represented_across_maps():
    maps = build_default_context_geometry_maps(num_depths=8)
    geometries = {g for spec in maps.values() for g in spec.geometry_by_depth}
    assert "hyperbolic" in geometries or "poincare" in geometries
    assert "spherical" in geometries
    assert "torus" in geometries
    assert "complex_projective" in geometries
    assert "subspace" in geometries
    assert "spatial_se3" in geometries
    assert "spcp" in geometries
    assert "holographic_phase" in geometries
