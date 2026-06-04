import torch

from mnemonic_cortex.reasoning_depth import (
    LTMDepthAdapter,
    LTMDepthAdapterConfig,
    MANNDepthAdapter,
    MANNDepthAdapterConfig,
    MANNLTMSharedSlotGeometry,
    SharedDepthSlotRegistry,
    SharedGeometrySlotConfig,
)


def _build_shared_geometry(dim: int = 16) -> MANNLTMSharedSlotGeometry:
    mann = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12))
    ltm = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12))
    return MANNLTMSharedSlotGeometry(
        config=SharedGeometrySlotConfig.enabled_default(key_dim=dim, value_dim=dim),
        mann_adapter=mann,
        ltm_adapter=ltm,
        registry=SharedDepthSlotRegistry(),
    )


def test_reason_shared_slot_runs_dual_geometry_and_links_registry():
    shared = _build_shared_geometry(dim=16)
    query = torch.randn(2, 4, 16)
    out, trace = shared.run_shared_reasoning(
        query,
        content="shared manifold slot",
        mann_slot_index=2,
        ltm_slot_index=5,
        hop_id=1,
        ltm_bank_name="cgmn_semantic",
        ltm_depth_index=5,
        mann_geometry_map="procedural",
        ltm_geometry_map="hierarchical",
        return_trace=True,
    )

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    assert trace["mann_geometry_map"] == "procedural"
    assert trace["ltm_geometry_map"] == "hierarchical"
    assert trace["mann_chart_transform"]["depth_index"] == 1
    assert trace["ltm_chart_transform"]["depth_index"] == 5
    assert trace["mann_ref"].startswith("mann.slot2.")
    assert trace["ltm_ref"].startswith("ltm.cgmn_semantic.slot5.")
    record = shared.registry.get(trace["canonical_slot_id"])
    assert record is not None
    assert trace["mann_ref"] in record.mann_refs
    assert trace["ltm_ref"] in record.ltm_refs
    assert record.to_dict()["safety"]["shared_physical_tensor"] is False


def test_reason_shared_slot_transforms_mann_and_ltm_slot_views_with_depth_geometry():
    shared = _build_shared_geometry(dim=16)
    transformed = shared.transform_slot_views(
        mann_slot_index=1,
        mann_depth_index=4,
        ltm_bank_name="hg_episodic",
        ltm_slot_index=3,
        ltm_depth_index=6,
        mann_geometry_map="quantum_holographic",
        ltm_geometry_map="procedural",
    )

    mann_tensor = torch.tensor(transformed["mann_slot_tensor"])
    ltm_tensor = torch.tensor(transformed["ltm_slot_tensor"])
    assert transformed["mann_slot_tensor_shape"] == [1, 16]
    assert transformed["ltm_slot_tensor_shape"] == [1, 16]
    assert torch.isfinite(mann_tensor).all()
    assert torch.isfinite(ltm_tensor).all()
    assert transformed["mann_chart_transform"]["geometry_map"] == "quantum_holographic"
    assert transformed["ltm_chart_transform"]["geometry_map"] == "procedural"
    assert transformed["mann_chart_transform"]["depth_index"] == 4
    assert transformed["ltm_chart_transform"]["depth_index"] == 6
    assert transformed["paamax_metadata"]["dual_geometry_maps_active"] is True
