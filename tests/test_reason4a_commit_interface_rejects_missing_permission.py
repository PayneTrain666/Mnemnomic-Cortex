from mnemonic_cortex.reasoning_depth import (
    ReasoningCommitInterface,
    ReasoningCommitInterfaceConfig,
    ReasoningCommitRequest,
    StoreOperationKind,
)


def test_reason4a_commit_interface_rejects_missing_permission():
    interface = ReasoningCommitInterface(
        ReasoningCommitInterfaceConfig(enabled=True, allow_explicit_commit=True)
    )
    request = ReasoningCommitRequest(
        target_store="ltm",
        payload={"proposal": "x"},
        operation_kind=StoreOperationKind.COMMIT,
        write_permission=False,
        dry_run=False,
    )
    decision = interface.decide(request).to_dict()

    assert decision["approved"] is False
    assert decision["status"] == "denied"
    assert "missing_write_permission" in decision["blocked_actions"]
    assert decision["safety_flags"]["real_store_write_performed"] is False
