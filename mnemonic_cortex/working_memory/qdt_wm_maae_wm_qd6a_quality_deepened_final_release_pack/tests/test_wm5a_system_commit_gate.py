import torch

from mnemonic_cortex.working_memory import (
    SystemCommitGate,
    SystemWriteProposal,
    SharedSlotStore,
    SharedSlotStoreConfig,
    QuantumHolographicStorage,
    QuantumHolographicStorageConfig,
)


def make_gate(dim=16, threshold=0.985):
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit_commit", dim=dim))
    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=dim, interference_threshold=threshold), shared_slot_store=store)
    gate = SystemCommitGate(dim=dim, shared_slot_store=store, qh_storage=qh, require_write_permission=True)
    return gate, store, qh


def test_commit_gate_commit_success_and_trace():
    gate, store, qh = make_gate()
    proposal = SystemWriteProposal.create(
        content=torch.randn(16),
        local_slot_id="slot_a",
        write_permission=True,
        confidence=0.9,
    )

    stage = gate.stage(proposal)
    evaluation = gate.evaluate(proposal.proposal_id)
    decision = gate.commit(proposal.proposal_id)

    assert stage["pending_count"] == 1
    assert evaluation.ok is True
    assert decision.decision == "commit"
    assert decision.canonical_slot_id is not None
    assert decision.qh_record_id is not None
    assert store.registry.to_dict()["record_count"] == 1
    assert qh.trace_summary()["record_count"] == 1
    assert gate.trace_summary()["decision_count"] == 1


def test_commit_gate_rejects_without_paamax_write_permission():
    gate, store, qh = make_gate()
    proposal = SystemWriteProposal.create(
        content=torch.randn(16),
        local_slot_id="slot_no_permission",
        write_permission=False,
        confidence=0.9,
    )

    gate.stage(proposal)
    evaluation = gate.evaluate(proposal.proposal_id)
    decision = gate.commit(proposal.proposal_id)

    assert evaluation.permission_ok is False
    assert decision.decision == "reject"
    assert "permission" in decision.reason
    assert store.registry.to_dict()["record_count"] == 0
    assert qh.trace_summary()["record_count"] == 0


def test_commit_gate_quarantines_interference():
    gate, store, qh = make_gate(threshold=0.95)
    content = torch.ones(16)

    first = SystemWriteProposal.create(content=content, local_slot_id="slot_one", write_permission=True, confidence=0.9)
    gate.stage(first)
    d1 = gate.commit(first.proposal_id)
    assert d1.decision == "commit"

    second = SystemWriteProposal.create(content=content.clone(), local_slot_id="slot_two", write_permission=True, confidence=0.9)
    gate.stage(second)
    evaluation = gate.evaluate(second.proposal_id)
    decision = gate.commit(second.proposal_id)

    assert evaluation.interference_ok is False
    assert decision.decision == "quarantine"
    assert decision.quarantine is True
    assert decision.canonical_slot_id is not None
    assert store.registry.get(decision.canonical_slot_id).conflict_state == "quarantined"


def test_commit_gate_rollback_last_commit():
    gate, store, qh = make_gate()
    proposal = SystemWriteProposal.create(content=torch.randn(16), local_slot_id="slot_rollback", write_permission=True, confidence=0.9)
    gate.stage(proposal)
    decision = gate.commit(proposal.proposal_id)
    assert decision.decision == "commit"
    before = store.registry.to_dict()["record_count"]
    assert before == 1

    rollback = gate.rollback_last("unit rollback")
    assert rollback.decision == "rollback"
    assert store.registry.to_dict()["record_count"] == 0
    assert qh.trace_summary()["record_count"] == 0


def test_commit_gate_rejects_unstable_content():
    gate, store, qh = make_gate()
    proposal = SystemWriteProposal.create(content=torch.ones(16) * 1e8, local_slot_id="slot_bad", write_permission=True, confidence=0.9)
    gate.stage(proposal)
    evaluation = gate.evaluate(proposal.proposal_id)
    decision = gate.commit(proposal.proposal_id)

    assert evaluation.stability_ok is False
    assert decision.decision == "reject"
    assert "stability" in decision.reason


def test_commit_gate_rejects_bad_shape():
    gate, _, _ = make_gate()
    proposal = SystemWriteProposal.create(content=torch.randn(2, 16), local_slot_id="bad", write_permission=True)
    try:
        gate.stage(proposal)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
