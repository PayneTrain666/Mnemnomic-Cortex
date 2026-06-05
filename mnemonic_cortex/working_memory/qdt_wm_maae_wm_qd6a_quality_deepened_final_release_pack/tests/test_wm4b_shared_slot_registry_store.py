import torch

from mnemonic_cortex.working_memory import (
    canonical_slot_id,
    SharedSlotRegistry,
    SharedSlotStore,
    SharedSlotStoreConfig,
    MirroredContentRule,
)


def test_canonical_slot_id_is_deterministic_and_namespaced():
    a = canonical_slot_id("ns", "slot-1", "fp")
    b = canonical_slot_id("ns", "slot-1", "fp")
    c = canonical_slot_id("ns", "slot-2", "fp")
    assert a == b
    assert a != c
    assert a.startswith("css-")


def test_shared_slot_registry_create_link_conflict_and_write_permission():
    registry = SharedSlotRegistry(namespace="unit")
    record = registry.get_or_create("ltm_slot_0001", memory_type="ltm", content_fingerprint="abc", geometry_map="hyperbolic")
    registry.link_mirror(record.canonical_id, memory_type="mann", local_slot_id="mann_slot_0001", geometry_map="subspace")
    registry.grant_write(record.canonical_id, True)
    registry.mark_conflict(record.canonical_id, "unit conflict", quarantine=True)

    out = registry.get(record.canonical_id)
    assert out is not None
    assert out.write_permission_granted is True
    assert out.conflict_state == "quarantined"
    assert sorted(out.source_memory_types) == ["ltm", "mann"]
    d = out.to_dict()
    assert len(d["mirrors"]) == 2


def test_shared_slot_store_write_mirror_trace_and_content():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit", dim=8))
    content = torch.randn(8)
    result = store.write_slot(
        memory_type="ltm",
        local_slot_id="ltm_slot_1",
        content=content,
        geometry_map="hyperbolic",
        confidence=0.9,
        write_permission=True,
    )
    store.mirror_slot(
        result.canonical_id,
        memory_type="mann",
        local_slot_id="mann_slot_1",
        geometry_map="subspace",
        confidence=0.8,
    )

    trace = store.trace_for_local_slot("mann", "mann_slot_1")
    assert result.canonical_id in trace["canonical_ids"]
    assert trace["paamax_metadata"]["write_permission_required"] is True
    recovered = store.get_content(result.canonical_id)
    assert torch.allclose(recovered, content)


def test_mirrored_content_rule_validation():
    rule = MirroredContentRule("ltm", "mann", mirror_mode="metadata_only")
    rule.validate()
    bad = MirroredContentRule("ltm", "mann", mirror_mode="content_copy", allow_content_copy=False)
    try:
        bad.validate()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
