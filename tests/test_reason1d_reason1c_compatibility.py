import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig, LTMDepthAdapter, LTMDepthAdapterConfig
from mnemonic_cortex.working_memory.ltm_depth_integration import install_optional_ltm_depth_adapter, ltm_reason1d_integration_contract


class Shell:
    pass


def test_reason1d_reason1c_mann_adapter_still_imports_and_works():
    mann = MANNDepthAdapter(MANNDepthAdapterConfig.disabled(key_dim=16))
    query = torch.randn(2, 3, 16)
    out, trace = mann.read_hop(query, hop_id=0, return_trace=True)
    assert out is query
    assert trace["pass_through"] is True


def test_reason1d_optional_ltm_adapter_install_is_non_destructive():
    shell = Shell()
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.disabled(key_dim=16))
    install_optional_ltm_depth_adapter(shell, adapter)
    assert hasattr(shell, "ltm_depth_adapter")
    assert shell.ltm_depth_adapter is adapter

    existing = shell.ltm_depth_adapter
    install_optional_ltm_depth_adapter(shell, LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16)))
    assert shell.ltm_depth_adapter is existing

    contract = ltm_reason1d_integration_contract()
    assert contract["destructive_ltm_replacement"] is False
    assert contract["shared_physical_tensor"] is False
