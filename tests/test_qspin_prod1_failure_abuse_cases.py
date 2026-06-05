from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
act = mods["qspin_runtime_shadow_activation"]
conf = mods["qspin_production_config"]
ks = mods["qspin_kill_switch"]
commit = mods["qspin_commit_gate_dry_run"]
rb = mods["qspin_rollback_harness"]


def assert_raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_malformed_request_rejected():
    assert_raises(lambda: act.QSpinRuntimeShadowActivationRequest("", source_matrix_complete=True).validate())


def test_unknown_kill_switch_state_fails_closed():
    assert_raises(lambda: ks.QSpinKillSwitch(state=ks.QSpinKillSwitchState.UNKNOWN).validate())


def test_attempted_write_and_raw_trace_block_in_commit_gate():
    result = commit.QSpinCommitGateDryRunInspector().inspect(commit.QSpinCommitGateDryRunRequest("abuse", source_matrix_complete=True, rollback_evidence_present=True, write_permissions_requested=True, raw_payload_trace_requested=True))
    assert result.decision.allowed_for_shadow is False


def test_shadow_result_cannot_claim_live_effects():
    kill = ks.build_default_qspin_kill_switch().decision()
    cg = commit.QSpinCommitGateDryRunInspector().inspect(commit.QSpinCommitGateDryRunRequest("cg", source_matrix_complete=True, rollback_evidence_present=True))
    evidence = tuple(rb.QSpinRollbackEvidenceRecord("ev_" + kind.value, kind) for kind in rb.QSpinRollbackEvidenceKind)
    rollback = rb.build_default_qspin_rollback_harness().dry_run(rb.QSpinRollbackDryRunRequest("rb", evidence))
    req = act.QSpinRuntimeShadowActivationRequest("abuse_result", source_matrix_complete=True)
    result = act.QSpinRuntimeShadowActivationController().evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert_raises(lambda: act.QSpinRuntimeShadowActivationResult(result.request, result.decision, result.trace, result.audit_event, routed_live_data=True).validate())
