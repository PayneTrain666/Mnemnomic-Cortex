import torch

from mnemonic_cortex.working_memory import QDTWorkingMemory, QDTWorkingMemoryConfig


def test_qdt_working_memory_write_uses_system_commit_gate():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="write", context_map_name="quantum_holographic", return_trace=True)

    assert y.shape == x.shape
    stages = [item["stage"] for item in trace["items"]]
    assert "system_commit_gate" in stages
    decisions = [
        item["metadata"]["decision"]
        for item in trace["items"]
        if item["stage"] == "system_commit_gate" and item["message"] == "write_decision"
    ]
    assert decisions
    assert decisions[-1]["decision"] in {"commit", "quarantine"}
    assert wm.system_commit_gate.trace_summary()["decision_count"] >= 1
    assert wm.shared_slot_store.registry.to_dict()["record_count"] >= 1
    assert wm.qh_storage.trace_summary()["record_count"] >= 1


def test_qdt_working_memory_read_exposes_commit_gate_summary():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="read", context_map_name="quantum_holographic", return_trace=True)

    assert y.shape == x.shape
    stages = [item["stage"] for item in trace["items"]]
    assert "system_commit_gate" in stages
    summaries = [
        item["metadata"]["gate_summary"]
        for item in trace["items"]
        if item["stage"] == "system_commit_gate" and item["message"] == "read_path_gate_summary"
    ]
    assert summaries
    assert summaries[-1]["trace_type"] == "system_commit_gate"


def test_qdt_working_memory_process_still_finite_after_commit_gate_patch():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y = wm(x, operation="process", context_map_name="quantum_holographic")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
