import json

from mnemonic_cortex.reasoning_depth import SharedDepthSlotRegistry, shared_depth_registry_contract


def test_reason1d_shared_registry_supports_provenance_and_refs_without_shared_tensors():
    reg = SharedDepthSlotRegistry()
    rec = reg.create_or_update(
        canonical_slot_id="concept.ltm.1",
        content="ltm content",
        wm_ref="wm.slot.1",
        mann_ref="mann.slot.2",
        ltm_ref="ltm.cgmn.3",
        source_stage="REASON-1D",
        source_pack="/tmp/source.zip",
        consolidation_status="shadow_proposed",
        registry_lineage=[{"event": "test"}],
        depth_roles_present=[1, 5],
    )
    payload = rec.to_dict()
    assert payload["source_stage"] == "REASON-1D"
    assert payload["source_pack"] == "/tmp/source.zip"
    assert payload["consolidation_status"] == "shadow_proposed"
    assert payload["safety"]["shared_physical_tensor"] is False
    assert "wm.slot.1" in payload["wm_refs"]
    assert "mann.slot.2" in payload["mann_refs"]
    assert "ltm.cgmn.3" in payload["ltm_refs"]

    contract = shared_depth_registry_contract()
    assert contract["shared_physical_tensor"] is False
    json.dumps(reg.to_dict())
