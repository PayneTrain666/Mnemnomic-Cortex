import json

from mnemonic_cortex.reasoning_depth import (
    ReasoningCommitInterface,
    ReasoningCommitInterfaceConfig,
    ReasoningCommitRequest,
    StoreOperationKind,
)


def test_reason4a_commit_interface_accepts_explicit_intent_metadata_only():
    interface = ReasoningCommitInterface(
        ReasoningCommitInterfaceConfig(enabled=True, allow_explicit_commit=True)
    )
    request = ReasoningCommitRequest(
        target_store="ltm",
        payload={"proposal": "x"},
        operation_kind=StoreOperationKind.COMMIT,
        write_permission=True,
        dry_run=False,
    )
    decision = interface.decide(request).to_dict()

    assert decision["approved"] is True
    assert decision["status"] == "approved_explicit_write_intent"
    assert decision["safety_flags"]["real_store_write_performed"] is False
    json.dumps(decision)
