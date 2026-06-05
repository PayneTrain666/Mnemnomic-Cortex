import json

from mnemonic_cortex.reasoning_depth import CommitDryRunLedger, CommitDryRunLedgerConfig


def test_reason4b_commit_dry_run_ledger_append():
    ledger = CommitDryRunLedger(CommitDryRunLedgerConfig.enabled_default())
    result = ledger.append(
        request={"idempotency_key": "k1", "payload": {"payload_id": "p1"}},
        decision={"approved": True, "status": "dry_run_only"},
        backend_result={"accepted": True},
    ).to_dict()
    payload = ledger.to_dict()

    assert result["accepted"] is True
    assert payload["record_count"] == 1
    assert payload["real_write_performed"] is False
    json.dumps(payload)
