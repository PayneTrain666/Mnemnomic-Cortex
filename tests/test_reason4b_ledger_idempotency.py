from mnemonic_cortex.reasoning_depth import CommitDryRunLedger, CommitDryRunLedgerConfig


def test_reason4b_ledger_rejects_duplicate_idempotency():
    ledger = CommitDryRunLedger(CommitDryRunLedgerConfig.enabled_default())
    first = ledger.append(request={"idempotency_key": "k1"}, decision={"approved": True}).to_dict()
    second = ledger.append(request={"idempotency_key": "k1"}, decision={"approved": True}).to_dict()

    assert first["accepted"] is True
    assert second["accepted"] is False
    assert "duplicate_idempotency_key" in second["blocked_actions"]
