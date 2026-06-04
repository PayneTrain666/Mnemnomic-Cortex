import torch

from mnemonic_cortex.reasoning_depth import WMDepthController, MANNDepthAdapter, MANNDepthAdapterConfig
from mnemonic_cortex.working_memory.mann_depth_integration import install_optional_mann_depth_adapter, mann_reason1c_integration_contract


class Shell:
    pass


def test_reason1c_reason1b_wm_controller_still_imports_and_works():
    wm = WMDepthController.disabled(input_dim=16)
    x = torch.randn(2, 3, 16)
    out, trace = wm.process_wm(x, return_trace=True)
    assert out is x
    assert trace["pass_through"] is True


def test_reason1c_optional_mann_depth_adapter_install_is_non_destructive():
    shell = Shell()
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.disabled(key_dim=16))
    install_optional_mann_depth_adapter(shell, adapter)
    assert hasattr(shell, "mann_depth_adapter")
    assert shell.mann_depth_adapter is adapter

    existing = shell.mann_depth_adapter
    install_optional_mann_depth_adapter(shell, MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16)))
    assert shell.mann_depth_adapter is existing

    contract = mann_reason1c_integration_contract()
    assert contract["destructive_mann_replacement"] is False
    assert contract["shared_physical_tensor_with_ltm"] is False
