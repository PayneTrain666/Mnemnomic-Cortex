import torch

from mnemonic_cortex.working_memory import QDTWorkingMemory, QDTWorkingMemoryConfig


def test_qdt_working_memory_read_creates_qh_trace_and_record():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="read", context_map_name="quantum_holographic", return_trace=True)

    assert y.shape == x.shape
    stages = [item["stage"] for item in trace["items"]]
    assert "quantum_holographic_storage" in stages
    qh_items = [item for item in trace["items"] if item["stage"] == "quantum_holographic_storage"]
    qh_payload = qh_items[-1]["metadata"]["record"]
    assert qh_payload["code_schema"]["depth_code"] == "depth-00"
    assert qh_payload["code_schema"]["task_mode_code"] == "task-qh"
    assert qh_payload["paamax_metadata"]["write_permission_granted"] is True
    assert wm.qh_storage.trace_summary()["record_count"] >= 1


def test_qdt_shared_slot_store_has_qh_refs_after_read():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    _, _ = wm(x, operation="read", context_map_name="quantum_holographic", return_trace=True)
    store_dict = wm.shared_slot_store.to_dict()
    assert store_dict["qh_record_refs"]
    found = False
    for refs in store_dict["qh_record_refs"].values():
        if refs:
            found = True
            assert refs[0]["composite_code"].startswith("qh-")
    assert found


def test_qdt_process_still_finite_with_qh_storage_enabled():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y = wm(x, operation="process", context_map_name="quantum_holographic")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
