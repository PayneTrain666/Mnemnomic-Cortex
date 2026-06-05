from tests.test_qspin_active_dry_run_executor import good_req
from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
ex = mods["qspin_active_dry_run_executor"]
sim = mods["qspin_commit_gate_approval_sim"]
conf = mods["qspin_production_config"]
rt = mods["qspin_payload_roundtrip_stub"]
perm = mods["qspin_permission_dry_run"]

def test_full_prod3_active_dry_run_chain_simulation_only():
    approval = sim.QSpinCommitGateApprovalSimulator().simulate(sim.QSpinCommitGateApprovalRequest(
        "approval", True, source_matrix_complete=True, rollback_evidence_present=True,
        kill_switch_state=conf.QSpinRuntimeKillSwitchState.ENABLED,
    ))
    assert approval.decision.approved_active_dry_run is True

    roundtrip = rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(
        "rt", rt.QSpinPayloadCodecStubKind.DENSE, "src", "dst", (2, 4), declared_bytes=32,
    ))
    assert roundtrip.decision.approved is True

    permission = perm.QSpinQHSharedSlotPermissionDryRun().check(perm.QSpinPermissionDryRunRequest(
        "perm", perm.QSpinPermissionScope.QH, perm.QSpinPermissionOperation.READ_SIMULATION, "qh",
        interference_check_present=True, commit_gate_review_present=True, qh_permission_metadata_present=True,
    ))
    assert permission.decision.approved is True

    result = ex.QSpinActiveDryRunBridgeExecutor().execute(good_req("integrated"))
    assert result.decision.executed_active_dry_run is True
    assert result.production_activated is False
