from mnemonic_cortex.reasoning_depth import (
    reasoning_release_candidate_contract,
    reasoning_api_freeze_contract,
    reasoning_regression_closure_contract,
)


def test_reason3d_contracts():
    assert reasoning_release_candidate_contract()["stage"] == "REASON-3D"
    assert reasoning_release_candidate_contract()["no_fake_production_complete_claim"] is True
    assert reasoning_api_freeze_contract()["api_breaking_changes_allowed"] is False
    assert reasoning_regression_closure_contract()["permanent_memory_store_mutation"] is False
