import torch

from mnemonic_cortex.reasoning_depth import WMDepthController


def test_reason1b_controller_disabled_and_enabled_modes():
    disabled = WMDepthController.disabled(input_dim=16)
    x = torch.randn(2, 16)
    y, trace = disabled.process_wm(x, return_trace=True)
    assert y is x
    assert trace["pass_through"] is True

    enabled = WMDepthController.enabled_default(input_dim=16, value_dim=16, slot_count=10)
    y2, trace2 = enabled.process_wm(x, return_trace=True)
    assert y2.shape == (2, 16)
    assert trace2["operation"] == "read"
    metrics = enabled.capacity_metrics()
    assert metrics["theoretical_capacity_multiplier"] == 8
