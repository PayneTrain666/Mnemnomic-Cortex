from tests.qspin_prod4_module_loader import load_prod4_modules
mods=load_prod4_modules(); h=mods["qspin_synthetic_payload_harness"]; sb=mods["qspin_synthetic_qh_shared_slot_sandbox"]; cg=mods["qspin_commit_gate_expanded_dry_run"]
from tests.test_qspin_synthetic_payload_harness import req

def test_production_activation_and_write_abuse_blocked():
    out=h.build_default_qspin_synthetic_payload_execution_harness().execute(req(production_activation_requested=True,write_requested=True))
    assert h.QSpinSyntheticPayloadBlockReason.PRODUCTION_ACTIVATION_REQUESTED in out.decision.block_reasons
    assert h.QSpinSyntheticPayloadBlockReason.WRITE_REQUESTED in out.decision.block_reasons

def test_raw_payload_in_sandbox_blocked():
    s=sb.build_default_qspin_synthetic_qh_shared_slot_sandbox()
    out=s.operate(sb.QSpinSyntheticSandboxRequest("raw",sb.QSpinSyntheticSandboxScope.QH,sb.QSpinSyntheticSandboxOperation.READ_METADATA,"cell",True,True,True,raw_payload={"secret":"no"}))
    assert sb.QSpinSyntheticSandboxBlockReason.RAW_PAYLOAD_PRESENT in out.decision.block_reasons

def test_expanded_gate_blocks_transfer_and_commit():
    evidence=tuple(cg.QSpinCommitGateEvidenceRecord(k,"ev_"+k.value) for k in cg.QSpinCommitGateEvidenceKind)
    out=cg.QSpinCommitGateExpandedApprovalDryRun().inspect(cg.QSpinCommitGateExpandedRequest("bad",evidence,real_payload_transfer_requested=True,commit_execution_requested=True))
    assert cg.QSpinCommitGateExpandedBlockReason.REAL_PAYLOAD_TRANSFER in out.decision.block_reasons
    assert cg.QSpinCommitGateExpandedBlockReason.COMMIT_EXECUTION in out.decision.block_reasons
