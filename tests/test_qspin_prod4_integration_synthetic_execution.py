from tests.test_qspin_synthetic_payload_harness import req
from tests.qspin_prod4_module_loader import load_prod4_modules
mods=load_prod4_modules(); h=mods["qspin_synthetic_payload_harness"]; sb=mods["qspin_synthetic_qh_shared_slot_sandbox"]; cg=mods["qspin_commit_gate_expanded_dry_run"]; rg=mods["qspin_runtime_safety_regression"]

def test_integration_synthetic_execution_flow():
    payload=h.build_default_qspin_synthetic_payload_execution_harness().execute(req())
    assert payload.decision.executed is True
    sandbox=sb.build_default_qspin_synthetic_qh_shared_slot_sandbox().operate(sb.QSpinSyntheticSandboxRequest("s",sb.QSpinSyntheticSandboxScope.QH,sb.QSpinSyntheticSandboxOperation.WRITE_SANDBOX,"cell",True,True,True))
    assert sandbox.decision.allowed is True
    evidence=tuple(cg.QSpinCommitGateEvidenceRecord(k,"ev_"+k.value) for k in cg.QSpinCommitGateEvidenceKind)
    gate=cg.QSpinCommitGateExpandedApprovalDryRun().inspect(cg.QSpinCommitGateExpandedRequest("g",evidence))
    assert gate.decision.approved is True
    suite=rg.build_default_qspin_runtime_safety_regression_suite().run()
    assert suite.failed_count == 0
