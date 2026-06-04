import json
import pytest

from mnemonic_cortex.reasoning_depth import (
    BackupRecoveryPlanner,
    BackupRecoveryConfig,
    BackendBackupRecoveryError,
)


def test_real_backend_a_backup_recovery_plan_no_real_execution():
    plan = BackupRecoveryPlanner(BackupRecoveryConfig.enabled_default()).build().to_dict()

    assert plan["enabled"] is True
    assert plan["real_backup_executed"] is False
    assert plan["real_restore_executed"] is False
    assert "real_backup_execution" in plan["blocked_actions"]
    json.dumps(plan)

    with pytest.raises(BackendBackupRecoveryError):
        BackupRecoveryConfig(enabled=True, allow_real_backup_execution=True).validate()
