from mnemonic_cortex.reasoning_depth import (
    planner_evaluation_contract,
    planner_failure_classifier_contract,
    planner_remediation_guidance_contract,
    planner_quality_hardening_contract,
    controller_planner_integration_contract,
    strategy_graph_persistence_readiness_contract,
)


def test_reason3c_reason3b_and_new_contracts():
    assert planner_evaluation_contract()["stage"] == "REASON-3B"
    assert planner_failure_classifier_contract()["automatic_patching"] is False
    assert planner_remediation_guidance_contract()["recommendation_only"] is True
    assert planner_quality_hardening_contract()["stage"] == "REASON-3C"
    assert controller_planner_integration_contract()["explicit_opt_in_required"] is True
    assert strategy_graph_persistence_readiness_contract()["automatic_persistence"] is False
