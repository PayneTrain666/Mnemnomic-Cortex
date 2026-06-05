import json

from mnemonic_cortex.reasoning_depth import create_default_dry_run_backend, BackendPayloadEnvelope


def test_real_backend_a_dry_run_write_accepts_without_real_write():
    interface = create_default_dry_run_backend()
    envelope = BackendPayloadEnvelope(
        payload_kind="strategy_graph",
        payload={"node_count": 2},
        idempotency_key="graph-1",
        audit_metadata={"stage": "REAL-BACKEND-IMPLEMENTATION-A"},
        redaction_status="not_required",
    )
    result = interface.dry_run_write(envelope).to_dict()

    assert result["accepted"] is True
    assert result["dry_run"] is True
    assert result["real_store_write_performed"] is False
    assert "real_store_write" in result["blocked_actions"]
    json.dumps(result)
