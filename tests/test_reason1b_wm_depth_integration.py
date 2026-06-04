from mnemonic_cortex.working_memory.wm_depth_integration import install_optional_wm_depth_controller, wm_qd_reason1b_integration_contract
from mnemonic_cortex.reasoning_depth import WMDepthController


class Shell:
    pass


def test_reason1b_optional_wm_depth_controller_attach_is_non_destructive():
    shell = Shell()
    install_optional_wm_depth_controller(shell, WMDepthController.disabled(input_dim=16))
    assert hasattr(shell, "wm_depth_controller")
    assert shell.wm_depth_controller.enabled is False

    existing = shell.wm_depth_controller
    install_optional_wm_depth_controller(shell, WMDepthController.enabled_default(input_dim=16, value_dim=16))
    assert shell.wm_depth_controller is existing


def test_reason1b_integration_contract_states_no_destructive_replacement():
    contract = wm_qd_reason1b_integration_contract()
    assert contract["destructive_qdt_replacement"] is False
    assert contract["default_enabled"] is False
