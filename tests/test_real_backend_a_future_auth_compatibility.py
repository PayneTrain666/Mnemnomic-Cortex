from mnemonic_cortex.reasoning_depth import (
    backend_authorization_contract,
    backend_implementation_plan_contract,
    backend_interface_protocol_contract,
    dry_run_backend_interface_contract,
    credential_scope_model_contract,
    backup_recovery_planning_contract,
    migration_dry_run_planning_contract,
)


def test_real_backend_a_future_auth_compatibility():
    assert backend_authorization_contract()["planning_only"] is True
    assert backend_implementation_plan_contract()["write_capable_code_generated"] is False
    assert backend_interface_protocol_contract()["real_writes_allowed"] is False
    assert dry_run_backend_interface_contract()["real_store_write_performed"] is False
    assert credential_scope_model_contract()["real_secret_loaded"] is False
    assert backup_recovery_planning_contract()["real_backup_executed"] is False
    assert migration_dry_run_planning_contract()["schema_migration_executed"] is False
