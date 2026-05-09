import torch

from mnemonic_cortex.working_memory import QDTWorkingMemoryConfig, QDTWorkingMemory


def test_qdt_working_memory_read_includes_dual_fusion_trace():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    stages = [item["stage"] for item in trace["items"]]
    assert "dual_fusion" in stages
    dual_items = [item for item in trace["items"] if item["stage"] == "dual_fusion"]
    payload = dual_items[-1]["metadata"]["payload"]
    assert payload["trace"]["paamax_metadata"]["mann_trace_visible"] is True
    assert payload["trace"]["mann_trace_visibility"]["scratchpad_tokens_shape"] == [2, 3, 32]
    assert "confidence" in payload["trace"]
    assert "disagreement" in payload["trace"]


def test_qdt_working_memory_process_is_finite_after_dual_fusion():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y = wm(x, operation="process")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
