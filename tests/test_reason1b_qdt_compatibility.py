import importlib
import torch

from mnemonic_cortex.reasoning_depth import WMDepthController


def test_reason1b_qdt_working_memory_still_imports_and_optional_controller_can_exist():
    module = importlib.import_module("mnemonic_cortex.working_memory.qdt_working_memory")
    assert module is not None

    # The depth controller remains separate and optional.
    controller = WMDepthController.disabled(input_dim=16)
    x = torch.randn(2, 3, 16)
    y, trace = controller.process_wm(x, return_trace=True)
    assert y is x
    assert trace["enabled"] is False
