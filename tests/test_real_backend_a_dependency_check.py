import json

from mnemonic_cortex.reasoning_depth import create_default_dry_run_backend


def test_real_backend_a_dependency_check_has_no_external_connection():
    report = create_default_dry_run_backend().dependency_check()

    assert report["requires_external_connection"] is False
    assert report["credentials_loaded"] is False
    assert report["schema_migration_executed"] is False
    assert report["real_store_write_performed"] is False
    json.dumps(report)
