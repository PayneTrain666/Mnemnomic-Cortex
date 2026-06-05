import json
import pytest

from mnemonic_cortex.reasoning_depth import (
    MigrationDryRunPlanner,
    MigrationDryRunConfig,
    MigrationDryRunError,
)


def test_real_backend_a_migration_dry_run_no_schema_execution():
    plan = MigrationDryRunPlanner(MigrationDryRunConfig.enabled_default()).build().to_dict()

    assert plan["enabled"] is True
    assert plan["schema_migration_executed"] is False
    assert "schema_migration_execution" in plan["blocked_actions"]
    json.dumps(plan)

    with pytest.raises(MigrationDryRunError):
        MigrationDryRunConfig(enabled=True, allow_schema_execution=True).validate()
