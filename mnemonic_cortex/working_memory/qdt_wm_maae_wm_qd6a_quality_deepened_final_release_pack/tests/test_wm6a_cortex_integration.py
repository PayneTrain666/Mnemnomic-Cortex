import torch

from mnemonic_cortex.working_memory import (
    CortexWorkingMemoryIntegrationConfig,
    EnhancedMnemonicCortexQDTAdapter,
    replace_cortex_working_memory,
    QDTWMCompatibilityWrapper,
    migration_patch_template,
)


class LegacyCortexShell:
    def __init__(self):
        self.working_memory = "legacy-wm"


def test_replace_cortex_working_memory_preserves_old_reference():
    cortex = LegacyCortexShell()
    cfg = CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)

    result = replace_cortex_working_memory(cortex, cfg)

    assert result.replaced is True
    assert isinstance(cortex.working_memory, QDTWMCompatibilityWrapper)
    assert cortex.legacy_working_memory == "legacy-wm"
    assert result.preserved_old_reference is True
    assert result.trace["dimensional_depth_preserved"]["num_depths"] == 8
    assert result.trace["dimensional_depth_preserved"]["system_commit_gate"] is True


def test_enhanced_mnemonic_cortex_qdt_adapter_routes_operations_and_trace():
    cfg = CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    cortex = EnhancedMnemonicCortexQDTAdapter(cfg)
    x = torch.randn(2, 5, 32)

    y_read = cortex(x, operation="read", context_map_name="quantum_holographic")
    assert y_read.shape == x.shape
    assert cortex.last_trace["operation"] == "read"

    y_process = cortex(x, operation="process", context_map_name="quantum_holographic")
    assert y_process.shape == x.shape
    assert cortex.last_trace["operation"] == "process"

    y_write = cortex(x, operation="write", context_map_name="quantum_holographic")
    assert y_write.shape == x.shape
    assert cortex.last_trace["operation"] == "write"
    stages = [item["stage"] for item in cortex.last_trace["qdt_trace"]["items"]]
    assert "system_commit_gate" in stages


def test_migration_patch_template_is_explicit_template_not_fake_patch():
    cfg = CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    template = migration_patch_template(cfg)

    assert "QDT-WM-MAAE WM-6A patch template" in template
    assert "replace_cortex_working_memory" in template
    assert "input_dim=32" in template


def test_cortex_adapter_rejects_bad_operation():
    cfg = CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    cortex = EnhancedMnemonicCortexQDTAdapter(cfg)
    x = torch.randn(2, 5, 32)
    try:
        cortex(x, operation="bad")
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
