import torch

from mnemonic_cortex.working_memory import (
    QDTWorkingMemory,
    QDTWorkingMemoryConfig,
    QDTWMCompatibilityConfig,
    QDTWMCompatibilityWrapper,
    CortexWorkingMemoryIntegrationConfig,
    EnhancedMnemonicCortexQDTAdapter,
    replace_cortex_working_memory,
    ensure_commit_decision_like,
)


class LegacyCortex:
    def __init__(self):
        self.working_memory = "legacy"


def test_qdt_write_path_commit_decision_validates_after_qd5a():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="write", context_map_name="quantum_holographic", return_trace=True)
    assert y.shape == x.shape
    decisions = [
        item["metadata"]["decision"]
        for item in trace["items"]
        if item["stage"] == "system_commit_gate" and item["message"] == "write_decision"
    ]
    assert decisions
    ensure_commit_decision_like("decision", decisions[-1])


def test_compatibility_wrapper_and_cortex_adapter_still_route_read_process_write():
    wrapper = QDTWMCompatibilityWrapper(QDTWMCompatibilityConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4))
    x = torch.randn(2, 5, 32)

    for method in ["read", "process", "write"]:
        y, trace = getattr(wrapper, method)(x, context_map_name="quantum_holographic", return_trace=True)
        assert y.shape == x.shape
        assert trace["operation"] == method

    cortex = EnhancedMnemonicCortexQDTAdapter(CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4))
    y = cortex(x, operation="write", context_map_name="quantum_holographic")
    assert y.shape == x.shape
    assert cortex.last_trace["operation"] == "write"


def test_replace_cortex_working_memory_preserves_legacy_reference_after_qd5a():
    cortex = LegacyCortex()
    result = replace_cortex_working_memory(
        cortex,
        CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4),
    )
    assert result.replaced is True
    assert cortex.legacy_working_memory == "legacy"
    assert hasattr(cortex.working_memory, "write")
