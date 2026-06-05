from mnemonic_cortex.reasoning_depth import reasoning_controller_contract, reasoning_policy_router_contract, depth_route_strategy_contract, confidence_disagreement_contract


def test_reason2b_contracts_and_reason2a_compatibility():
    controller_contract = reasoning_controller_contract()
    assert controller_contract["default_enabled"] is False
    assert controller_contract["optional_policy_router"] is True
    assert controller_contract["canonical_slot_prefix_compatibility"] == "reason2a."
    assert reasoning_policy_router_contract()["default_enabled"] is False
    assert depth_route_strategy_contract()["bounded_hops"] is True
    assert confidence_disagreement_contract()["bounded_scores"] is True
