from tests.test_qspin_runtime_shadow_activation import safe_inputs
from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
act = mods["qspin_runtime_shadow_activation"]


def test_config_kill_switch_commit_dry_run_and_rollback_combine_to_shadow_only():
    kill, cg, rollback = safe_inputs()
    req = act.QSpinRuntimeShadowActivationRequest("integration", source_matrix_complete=True)
    result = act.QSpinRuntimeShadowActivationController().evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert result.decision.allowed_shadow_only is True
    assert result.production_activated is False
    assert result.routed_live_data is False
