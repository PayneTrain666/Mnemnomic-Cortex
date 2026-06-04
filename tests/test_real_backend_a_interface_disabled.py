from mnemonic_cortex.reasoning_depth import DryRunBackendInterface, BackendInterfaceConfig, BackendPayloadEnvelope


def test_real_backend_a_interface_disabled_rejects_dry_run():
    interface = DryRunBackendInterface(BackendInterfaceConfig.disabled())
    envelope = BackendPayloadEnvelope(payload_kind="trace", payload={"x": 1}, idempotency_key="k1")
    result = interface.dry_run_write(envelope).to_dict()

    assert result["accepted"] is False
    assert result["dry_run"] is True
    assert result["real_store_write_performed"] is False
    assert "backend_interface_disabled" in result["blocked_actions"]
