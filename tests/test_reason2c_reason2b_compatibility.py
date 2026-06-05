from mnemonic_cortex.reasoning_depth import (
    reasoning_controller_contract,
    reasoning_policy_router_contract,
    evidence_reasoning_contract,
    counterfactual_reasoning_contract,
    conflict_aware_consolidation_contract,
)


def test_reason2c_contracts_and_reason2b_compatibility():
    contract = reasoning_controller_contract()
    assert contract["default_enabled"] is False
    assert contract["optional_policy_router"] is True
    assert contract["optional_evidence_reasoning"] is True
    assert contract["canonical_slot_prefix_compatibility"] == "reason2a."
    assert reasoning_policy_router_contract()["default_enabled"] is False
    assert evidence_reasoning_contract()["default_enabled"] is False
    assert counterfactual_reasoning_contract()["no_real_ablation_execution"] is True
    assert conflict_aware_consolidation_contract()["no_commit_by_default"] is True
