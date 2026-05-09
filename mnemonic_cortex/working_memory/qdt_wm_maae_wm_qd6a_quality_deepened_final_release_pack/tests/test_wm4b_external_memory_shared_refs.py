import torch

from mnemonic_cortex.working_memory import (
    ExternalMemoryQuery,
    SyntheticExternalMemoryBank,
    SharedSlotStore,
    SharedSlotStoreConfig,
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
)


def test_synthetic_external_memory_bank_emits_shared_slot_refs():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit", dim=16))
    bank = SyntheticExternalMemoryBank("ltm", dim=16, slots=8, shared_slot_store=store)
    query = ExternalMemoryQuery("ltm", query_state=torch.randn(2, 16), metadata={"geometry_map": "hyperbolic"})
    response = bank.query(query, top_k=3)

    assert response.shared_slot_refs is not None
    assert len(response.shared_slot_refs) == 2
    assert len(response.shared_slot_refs[0]) == 3
    assert store.registry.to_dict()["record_count"] >= 3
    d = response.to_dict()
    assert d["shared_slot_refs"] == response.shared_slot_refs
    assert response.trace["shared_slot_refs"] == response.shared_slot_refs


def test_qdt_working_memory_dual_fusion_updates_shared_slot_registry_trace():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    stages = [item["stage"] for item in trace["items"]]
    assert "shared_slot_store" in stages
    assert wm.shared_slot_store.registry.to_dict()["record_count"] > 0

    shared_items = [item for item in trace["items"] if item["stage"] == "shared_slot_store"]
    registry = shared_items[-1]["metadata"]["registry"]
    assert registry["trace_type"] == "shared_slot_registry"
    assert registry["record_count"] > 0


def test_ltm_and_mann_shared_references_can_coexist_for_same_store():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit2", dim=16))
    ltm = SyntheticExternalMemoryBank("ltm", dim=16, slots=8, shared_slot_store=store)
    mann = SyntheticExternalMemoryBank("mann", dim=16, slots=8, shared_slot_store=store)
    query_ltm = ExternalMemoryQuery("ltm", query_state=torch.randn(1, 16))
    query_mann = ExternalMemoryQuery("mann", query_state=torch.randn(1, 16))

    ltm_resp = ltm.query(query_ltm, top_k=2)
    mann_resp = mann.query(query_mann, top_k=2)

    assert ltm_resp.shared_slot_refs is not None
    assert mann_resp.shared_slot_refs is not None
    assert store.references_for_memory("ltm")
    assert store.references_for_memory("mann")
