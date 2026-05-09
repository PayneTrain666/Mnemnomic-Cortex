import torch

from mnemonic_cortex.working_memory import (
    QDTWMCompatibilityConfig,
    QDTWMCompatibilityWrapper,
)


def test_compatibility_wrapper_forward_process_read_write_shapes_and_trace():
    wrapper = QDTWMCompatibilityWrapper(
        QDTWMCompatibilityConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    )
    x = torch.randn(2, 5, 32)

    y_process, trace_process = wrapper.process(x, return_trace=True, context_map_name="quantum_holographic")
    y_read, trace_read = wrapper.read(x, return_trace=True, context_map_name="quantum_holographic")
    y_write, trace_write = wrapper.write(x, return_trace=True, context_map_name="quantum_holographic")

    assert y_process.shape == x.shape
    assert y_read.shape == x.shape
    assert y_write.shape == x.shape
    assert trace_process["operation"] == "process"
    assert trace_read["operation"] == "read"
    assert trace_write["operation"] == "write"
    assert trace_write["routed_to"] == "QDTWorkingMemory"
    assert "system_commit_gate" in [item["stage"] for item in trace_write["qdt_trace"]["items"]]


def test_compatibility_wrapper_default_forward_and_stability_report():
    wrapper = QDTWMCompatibilityWrapper(
        QDTWMCompatibilityConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4, default_operation="process")
    )
    x = torch.randn(2, 5, 32)

    y = wrapper(x, context_map_name="quantum_holographic")
    report = wrapper.stability_report(x)

    assert y.shape == x.shape
    assert report["ok"] is True
    assert report["wrapper"] == "QDTWMCompatibilityWrapper"


def test_compatibility_wrapper_rejects_bad_shape():
    wrapper = QDTWMCompatibilityWrapper(QDTWMCompatibilityConfig(input_dim=32, hidden_dim=64, num_heads=4))
    bad = torch.randn(2, 32)
    try:
        wrapper(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
