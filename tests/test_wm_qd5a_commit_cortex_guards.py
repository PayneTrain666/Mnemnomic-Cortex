import json
import torch
import pytest

from mnemonic_cortex.working_memory import (
    WMCommitCortexValidationError,
    SystemWriteProposal,
    CommitGateDecision,
    ensure_commit_proposal_like,
    ensure_commit_decision_like,
    ensure_rollback_trace,
    ensure_compatibility_input,
    ensure_migration_template_safety,
    ensure_no_fake_real_source_patch_claim,
    commit_cortex_trace,
    commit_cortex_contract_trace,
    CortexWorkingMemoryIntegrationConfig,
    migration_patch_template,
)


def test_commit_proposal_and_decision_validation():
    proposal = SystemWriteProposal.create(
        content=torch.randn(32),
        local_slot_id="unit",
        write_permission=True,
        confidence=0.9,
    )
    ensure_commit_proposal_like("proposal", proposal, expected_dim=32)

    decision = CommitGateDecision(
        proposal_id=proposal.proposal_id,
        decision="commit",
        reason="unit",
        paamax_metadata={"decision": "commit"},
    )
    ensure_commit_decision_like("decision", decision)

    bad = SystemWriteProposal.create(content=torch.randn(32), write_permission=True, confidence=0.9)
    bad.content[0] = float("nan")
    with pytest.raises(WMCommitCortexValidationError):
        ensure_commit_proposal_like("bad", bad, expected_dim=32)


def test_rollback_trace_and_compatibility_input_validation():
    trace = {
        "decision": "rollback",
        "reason": "unit",
        "automatic_memory_store_mutation": False,
        "paamax_metadata": {"decision": "rollback"},
    }
    ensure_rollback_trace("trace", trace)

    x = torch.randn(2, 5, 32)
    ensure_compatibility_input("x", x, expected_dim=32)

    with pytest.raises(WMCommitCortexValidationError):
        ensure_compatibility_input("bad", torch.randn(2, 32), expected_dim=32)


def test_migration_template_safety_and_no_fake_real_source_claim():
    cfg = CortexWorkingMemoryIntegrationConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    template = migration_patch_template(cfg)
    ensure_migration_template_safety("template", template)

    ensure_no_fake_real_source_patch_claim("payload", {"real_cortex_source_available": False, "real_cortex_source_patched": False})
    with pytest.raises(WMCommitCortexValidationError):
        ensure_no_fake_real_source_patch_claim("bad_payload", {"real_cortex_source_available": False, "real_cortex_source_patched": True})


def test_commit_cortex_traces_are_json_safe_and_paamax_compatible():
    trace = commit_cortex_trace(module="unit", message="ok", payload={"x": torch.randn(2, 2)})
    contract = commit_cortex_contract_trace(module="unit")
    json.dumps(trace)
    json.dumps(contract)
    assert trace["paamax_metadata"]["commit_cortex_contract"] is True
    assert contract["payload"]["allowed_commit_decisions"] == ["commit", "reject", "rollback", "quarantine"]
