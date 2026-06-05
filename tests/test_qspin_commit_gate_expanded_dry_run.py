from tests.qspin_prod4_module_loader import load_prod4_modules
m=load_prod4_modules()["qspin_commit_gate_expanded_dry_run"]

def evidence(exclude=()):
    return tuple(m.QSpinCommitGateEvidenceRecord(k,"ev_"+k.value) for k in m.QSpinCommitGateEvidenceKind if k not in exclude)

def test_all_evidence_approves_dry_run_only():
    out=m.QSpinCommitGateExpandedApprovalDryRun().inspect(m.QSpinCommitGateExpandedRequest("ok",evidence()))
    assert out.decision.approved is True
    assert out.commit_executed is False and out.runtime_activated is False

def test_missing_evidence_and_unsafe_intents_block():
    out=m.QSpinCommitGateExpandedApprovalDryRun().inspect(m.QSpinCommitGateExpandedRequest("bad",evidence({m.QSpinCommitGateEvidenceKind.QD6A_MATRIX}),write_intent=True,raw_trace_intent=True,production_activation_requested=True))
    assert out.decision.approved is False
    assert m.QSpinCommitGateExpandedBlockReason.MISSING_EVIDENCE in out.decision.block_reasons
    assert m.QSpinCommitGateExpandedBlockReason.WRITE_INTENT in out.decision.block_reasons
    assert m.QSpinCommitGateEvidenceKind.QD6A_MATRIX in out.decision.missing_evidence
