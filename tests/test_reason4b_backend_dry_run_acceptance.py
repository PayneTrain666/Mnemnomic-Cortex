import json

from mnemonic_cortex.reasoning_depth import PersistenceBackendConfig, PersistenceBackendStub


def test_reason4b_backend_dry_run_acceptance():
    backend = PersistenceBackendStub(PersistenceBackendConfig.enabled_default())
    result = backend.dry_run_write({"payload_id": "p1", "items": [{"id": "x"}]}).to_dict()

    assert result["accepted"] is True
    assert result["reason"] == "dry-run accepted; no real write performed"
    assert result["real_write_performed"] is False
    json.dumps(result)
