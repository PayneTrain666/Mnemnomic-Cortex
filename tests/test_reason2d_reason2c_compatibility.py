from mnemonic_cortex.reasoning_depth import (
    reasoning_controller_contract,
    evidence_reasoning_contract,
    counterfactual_reasoning_contract,
    conflict_aware_consolidation_contract,
    reasoning_controller_api_contract,
    reasoning_release_audit_contract,
    reasoning_regression_matrix_contract,
)


def test_reason2d_reason2c_contract_compatibility():
    assert reasoning_controller_contract()["stage"] == "REASON-2C"
    assert reasoning_controller_contract()["canonical_slot_prefix_compatibility"] == "reason2a."
    assert evidence_reasoning_contract()["default_enabled"] is False
    assert counterfactual_reasoning_contract()["no_real_ablation_execution"] is True
    assert conflict_aware_consolidation_contract()["no_commit_by_default"] is True
    assert reasoning_controller_api_contract()["stage"] == "REASON-2D"
    assert reasoning_release_audit_contract()["checks_disabled_default"] is True
    assert reasoning_regression_matrix_contract()["json_safe_export"] is True
