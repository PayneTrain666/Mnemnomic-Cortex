import torch

from mnemonic_cortex.working_memory import QDTWorkingMemory, QDTWorkingMemoryConfig


def test_qdt_working_memory_attention_path_still_runs_read_process_write_after_qd3a():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    for op in ["read", "process", "write"]:
        y, trace = wm(x, operation=op, context_map_name="quantum_holographic", return_trace=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert "items" in trace
        assert len(trace["items"]) > 0


def test_qdt_working_memory_read_trace_contains_attention_or_fusion_metadata():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    _, trace = wm(x, operation="read", context_map_name="quantum_holographic", return_trace=True)
    stages = {item["stage"] for item in trace["items"]}
    assert "dual_fusion" in stages or "memory_augmented_attention" in stages
