from mnemonic_cortex.reasoning_depth import (
    reasoning_persistence_backends_contract,
    reasoning_commit_dry_run_ledger_contract,
    reasoning_persistence_recovery_contract,
    reasoning_persistence_line_closure_contract,
    reasoning_integration_index_contract,
    reasoning_final_safety_audit_contract,
)


def test_reason4c_reason4b_compatibility_and_contracts():
    assert reasoning_persistence_backends_contract()["stage"] == "REASON-4B"
    assert reasoning_commit_dry_run_ledger_contract()["dry_run_only"] is True
    assert reasoning_persistence_recovery_contract()["real_rollback_performed"] is False
    assert reasoning_persistence_line_closure_contract()["stage"] == "REASON-4C"
    assert reasoning_integration_index_contract()["covers_reason_1a_to_4c"] is True
    assert reasoning_final_safety_audit_contract()["future_backend_requires_explicit_authorization"] is True
