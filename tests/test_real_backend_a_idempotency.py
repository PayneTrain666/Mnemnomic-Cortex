from mnemonic_cortex.reasoning_depth import create_default_dry_run_backend, BackendPayloadEnvelope


def test_real_backend_a_idempotency_duplicate_rejected():
    interface = create_default_dry_run_backend()
    envelope = BackendPayloadEnvelope(payload_kind="trace", payload={"x": 1}, idempotency_key="dup-key")

    first = interface.dry_run_write(envelope).to_dict()
    second = interface.dry_run_write(envelope).to_dict()

    assert first["accepted"] is True
    assert second["accepted"] is False
    assert "duplicate_idempotency_key" in second["blocked_actions"]
