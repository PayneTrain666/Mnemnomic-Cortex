from mnemonic_cortex.reasoning_depth import (
    reasoning_persistence_adapter_contract,
    reasoning_commit_interface_contract,
    reasoning_store_safety_contracts_contract,
    reasoning_persistence_backends_contract,
    reasoning_commit_dry_run_ledger_contract,
    reasoning_persistence_recovery_contract,
)


def test_reason4b_reason4a_compatibility_and_contracts():
    assert reasoning_persistence_adapter_contract()["stage"] == "REASON-4A"
    assert reasoning_commit_interface_contract()["real_store_write_performed"] is False
    assert reasoning_store_safety_contracts_contract()["automatic_persistence"] is False
    assert reasoning_persistence_backends_contract()["stage"] == "REASON-4B"
    assert reasoning_commit_dry_run_ledger_contract()["dry_run_only"] is True
    assert reasoning_persistence_recovery_contract()["real_rollback_performed"] is False
