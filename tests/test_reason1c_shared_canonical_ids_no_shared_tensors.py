import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig, SharedDepthSlotRegistry


def test_reason1c_mann_ltm_share_canonical_ids_not_physical_tensors():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=12))
    registry = SharedDepthSlotRegistry()

    registry.create_or_update(
        canonical_slot_id="concept.shared.1",
        content="shared semantic concept",
        mann_ref="mann.slot.3",
        ltm_ref="ltm.cgmn.42",
        depth_roles_present=[1, 2, 4],
    )

    proposal_result = adapter.propose_hop_memory(
        slot_index=3,
        key=torch.randn(16),
        value=torch.randn(16),
        canonical_slot_id="concept.shared.1",
        hop_id=0,
    )

    record = registry.get("concept.shared.1")
    assert record is not None
    assert "mann.slot.3" in record.mann_refs
    assert "ltm.cgmn.42" in record.ltm_refs
    assert record.to_dict()["safety"]["shared_physical_tensor"] is False
    assert proposal_result["metadata"]["no_shared_physical_tensor_with_ltm"] is True
