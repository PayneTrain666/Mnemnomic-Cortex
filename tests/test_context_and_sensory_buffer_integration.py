import torch

from mnemonic_cortex.sensory_buffer import EnhancedSensoryBuffer
from mnemonic_cortex.working_memory import QDTWorkingMemory, QDTWorkingMemoryConfig


def test_sensory_buffer_head_selection_supports_non_multiple_of_8():
    buf = EnhancedSensoryBuffer(buffer_size=4, input_dim=30)
    assert int(buf.num_heads) == 2


def test_sensory_buffer_filter_shape_and_finite_output():
    torch.manual_seed(0)
    buf = EnhancedSensoryBuffer(buffer_size=4, input_dim=16)
    x = torch.randn(2, 5, 16)
    buf.update(x)
    y = buf.attention_filter(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_qdt_working_memory_context_buffer_mounts_with_trace():
    torch.manual_seed(0)
    wm = QDTWorkingMemory(
        QDTWorkingMemoryConfig(
            input_dim=16,
            hidden_dim=32,
            num_depths=8,
            num_slots=8,
            num_heads=4,
        )
    )
    x = torch.randn(2, 4, 16)
    context = torch.randn(2, 3, 16)
    _, trace = wm(
        x,
        operation="read",
        context=context,
        context_map_name="quantum_holographic",
        return_trace=True,
    )
    items = trace.get("items", [])
    context_items = [it for it in items if it.get("stage") == "context_buffer"]
    assert len(context_items) >= 1
