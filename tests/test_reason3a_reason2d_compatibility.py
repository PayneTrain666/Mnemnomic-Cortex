from mnemonic_cortex.reasoning_depth import (
    reasoning_controller_api_contract,
    reasoning_release_audit_contract,
    reasoning_regression_matrix_contract,
    reasoning_strategy_graph_contract,
    multi_pass_thought_planner_contract,
    evidence_guided_route_expander_contract,
)


def test_reason3a_reason2d_and_new_contracts():
    assert reasoning_controller_api_contract()["stage"] == "REASON-2D"
    assert reasoning_release_audit_contract()["checks_disabled_default"] is True
    assert reasoning_regression_matrix_contract()["json_safe_export"] is True
    assert reasoning_strategy_graph_contract()["stage"] == "REASON-3A"
    assert multi_pass_thought_planner_contract()["default_enabled"] is False
    assert evidence_guided_route_expander_contract()["permanent_memory_store_mutation"] is False
