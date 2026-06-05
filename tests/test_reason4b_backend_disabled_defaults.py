from mnemonic_cortex.reasoning_depth import PersistenceBackendConfig, PersistenceBackendStub


def test_reason4b_backend_disabled_defaults():
    backend = PersistenceBackendStub(PersistenceBackendConfig.disabled())
    report = backend.dependency_check().to_dict()
    result = backend.dry_run_write({"payload_id": "p1", "items": []}).to_dict()

    assert report["status"] == "disabled"
    assert report["checks"]["allow_real_writes"] is False
    assert result["accepted"] is False
    assert result["real_write_performed"] is False
