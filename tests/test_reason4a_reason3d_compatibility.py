from mnemonic_cortex.reasoning_depth import (
    reasoning_release_candidate_contract,
    reasoning_api_freeze_contract,
    reasoning_regression_closure_contract,
    reasoning_persistence_adapter_contract,
    reasoning_commit_interface_contract,
    reasoning_store_safety_contracts_contract,
)


def test_reason4a_reason3d_compatibility_and_contracts():
    assert reasoning_release_candidate_contract()["stage"] == "REASON-3D"
    assert reasoning_api_freeze_contract()["api_breaking_changes_allowed"] is False
    assert reasoning_regression_closure_contract()["stage"] == "REASON-3D"
    assert reasoning_persistence_adapter_contract()["stage"] == "REASON-4A"
    assert reasoning_commit_interface_contract()["real_store_write_performed"] is False
    assert reasoning_store_safety_contracts_contract()["automatic_persistence"] is False
