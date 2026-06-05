from mnemonic_cortex.reasoning_depth import (
    planner_quality_hardening_contract,
    controller_planner_integration_contract,
    strategy_graph_persistence_readiness_contract,
)


def test_reason3d_reason3c_compatibility():
    assert planner_quality_hardening_contract()["stage"] == "REASON-3C"
    assert controller_planner_integration_contract()["explicit_opt_in_required"] is True
    assert strategy_graph_persistence_readiness_contract()["automatic_persistence"] is False
