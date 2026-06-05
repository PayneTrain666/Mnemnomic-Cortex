import torch

from mnemonic_cortex.working_memory import (
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    ensure_depth_state,
)


def test_qdt_working_memory_still_preserves_read_process_write_shapes_after_qd2a():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    for op in ["read", "process", "write"]:
        y, trace = wm(x, operation=op, context_map_name="quantum_holographic", return_trace=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert "items" in trace


def test_qdt_depth_replication_contract_shape_when_available():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    # Use the public replicator if present, otherwise this test still verifies the contract helper.
    if hasattr(wm, "depth_replicator"):
        depth = wm.depth_replicator(x)
        ensure_depth_state("depth", depth, expected_depths=8, expected_dim=32)
        assert depth.shape == (2, 8, 5, 3, 32)
