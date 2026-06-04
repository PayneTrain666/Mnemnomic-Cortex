from mnemonic_cortex.reasoning_depth import (
    reasoning_strategy_graph_contract,
    multi_pass_thought_planner_contract,
    evidence_guided_route_expander_contract,
    planner_evaluation_contract,
    planner_failure_classifier_contract,
    planner_remediation_guidance_contract,
)


def test_reason3b_reason3a_and_new_contracts():
    assert reasoning_strategy_graph_contract()["stage"] == "REASON-3A"
    assert multi_pass_thought_planner_contract()["default_enabled"] is False
    assert evidence_guided_route_expander_contract()["permanent_memory_store_mutation"] is False
    assert planner_evaluation_contract()["stage"] == "REASON-3B"
    assert planner_failure_classifier_contract()["automatic_patching"] is False
    assert planner_remediation_guidance_contract()["recommendation_only"] is True
