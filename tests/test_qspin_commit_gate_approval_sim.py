from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
sim = mods["qspin_commit_gate_approval_sim"]
conf = mods["qspin_production_config"]

def test_commit_gate_approval_active_dry_run_only():
    req = sim.QSpinCommitGateApprovalRequest(
        "ok", active_dry_run_scope_declared=True,
        source_matrix_complete=True, rollback_evidence_present=True,
        kill_switch_state=conf.QSpinRuntimeKillSwitchState.ENABLED,
    )
    result = sim.QSpinCommitGateApprovalSimulator().simulate(req)
    assert result.decision.approved_active_dry_run is True
    assert result.commit_executed is False
    assert result.production_activated is False

def test_commit_gate_blocks_production_activation_and_unsafe_flags():
    req = sim.QSpinCommitGateApprovalRequest(
        "bad", active_dry_run_scope_declared=True,
        requested_feature_flags=frozenset({conf.QSpinRuntimeFeatureFlag.PRODUCTION_ACTIVATION}),
        source_matrix_complete=True, rollback_evidence_present=True,
        kill_switch_state=conf.QSpinRuntimeKillSwitchState.ENABLED,
        production_activation_requested=True,
        commit_execution_requested=True,
    )
    result = sim.QSpinCommitGateApprovalSimulator().simulate(req)
    reasons = set(result.decision.block_reasons)
    assert sim.QSpinCommitGateApprovalSimBlockReason.UNSAFE_RUNTIME_FLAG in reasons
    assert sim.QSpinCommitGateApprovalSimBlockReason.PRODUCTION_ACTIVATION_REQUESTED in reasons
    assert sim.QSpinCommitGateApprovalSimBlockReason.COMMIT_EXECUTION_REQUESTED in reasons

def test_commit_gate_blocks_missing_matrix_rollback_and_kill_switch():
    req = sim.QSpinCommitGateApprovalRequest(
        "missing", active_dry_run_scope_declared=False,
        source_matrix_complete=False, rollback_evidence_present=False,
        kill_switch_state=conf.QSpinRuntimeKillSwitchState.TRIPPED,
    )
    result = sim.QSpinCommitGateApprovalSimulator().simulate(req)
    reasons = set(result.decision.block_reasons)
    assert sim.QSpinCommitGateApprovalSimBlockReason.ACTIVE_DRY_RUN_SCOPE_MISSING in reasons
    assert sim.QSpinCommitGateApprovalSimBlockReason.SOURCE_MATRIX_MISSING in reasons
    assert sim.QSpinCommitGateApprovalSimBlockReason.ROLLBACK_EVIDENCE_MISSING in reasons
    assert sim.QSpinCommitGateApprovalSimBlockReason.KILL_SWITCH_NOT_ENABLED in reasons
