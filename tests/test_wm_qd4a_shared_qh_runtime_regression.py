import torch

from mnemonic_cortex.working_memory import (
    SharedSlotStore,
    SharedSlotStoreConfig,
    QuantumHolographicStorage,
    QuantumHolographicStorageConfig,
    ensure_shared_slot_record,
    ensure_qh_storage_record,
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
)


def test_shared_slot_store_and_qh_storage_records_validate_after_qd4a():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="qd4a", dim=16))
    content = torch.randn(16)
    write = store.write_slot(
        memory_type="ltm",
        local_slot_id="slot",
        content=content,
        owner="ltm",
        geometry_map="holographic_phase",
        depth_index=0,
        confidence=0.9,
        write_permission=True,
    )
    record = store.registry.get(write.canonical_id)
    ensure_shared_slot_record("shared_record", record.to_dict())

    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=16), shared_slot_store=store)
    qh_record = qh.create_from_shared_slot(
        canonical_slot_id=write.canonical_id,
        depth_index=0,
        bank_name="ltm",
        geometry_name="holographic_phase",
        triplet_index=0,
        memory_type="ltm",
        task_mode="quantum_holographic",
        confidence=0.9,
        write_permission=True,
    )
    ensure_qh_storage_record("qh_record", qh_record.to_dict())


def test_qdt_working_memory_still_runs_with_shared_slot_qh_external_layers():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    for op in ["read", "process", "write"]:
        y, trace = wm(x, operation=op, context_map_name="quantum_holographic", return_trace=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert "items" in trace

    assert wm.shared_slot_store.registry.to_dict()["record_count"] >= 1
    assert wm.qh_storage.trace_summary()["record_count"] >= 1
