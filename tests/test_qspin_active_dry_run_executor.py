from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
ex = mods["qspin_active_dry_run_executor"]

def good_req(request_id="exec_ok"):
    return ex.QSpinActiveDryRunExecutionRequest(
        request_id=request_id,
        bridge_plan_id="plan",
        feature_gate=ex.QSpinActiveDryRunFeatureGate(active_dry_run_enabled=True),
        source_matrix_complete=True,
        rollback_evidence_present=True,
        kill_switch_allows=True,
        commit_approval_allows=True,
        payload_roundtrip_approved=True,
        permission_dry_run_approved=True,
        guarded_dispatch_approved=True,
        trace_safe_payload_summary_present=True,
    )

def test_active_dry_run_executor_allows_simulation_only():
    result = ex.QSpinActiveDryRunBridgeExecutor().execute(good_req())
    assert result.decision.executed_active_dry_run is True
    assert result.called_live_runtime is False
    assert result.routed_live_data is False
    assert result.transferred_payload is False
    assert result.wrote_state is False
    assert result.executed_commit is False
    assert result.production_activated is False

def test_active_dry_run_executor_idempotent():
    runner = ex.QSpinActiveDryRunBridgeExecutor()
    req = good_req("same")
    first = runner.execute(req)
    second = runner.execute(req)
    assert first.decision.status is ex.QSpinActiveDryRunStatus.EXECUTED_ACTIVE_DRY_RUN
    assert second.decision.status is ex.QSpinActiveDryRunStatus.IDEMPOTENT_REPLAY

def test_active_dry_run_blocks_missing_gates_and_live_effects():
    req = ex.QSpinActiveDryRunExecutionRequest(
        "bad", "plan", ex.QSpinActiveDryRunFeatureGate(active_dry_run_enabled=False),
        source_matrix_complete=False, rollback_evidence_present=False, kill_switch_allows=False,
        commit_approval_allows=False, payload_roundtrip_approved=False, permission_dry_run_approved=False,
        guarded_dispatch_approved=False, trace_safe_payload_summary_present=False,
        live_routing_requested=True, payload_transfer_requested=True, write_requested=True,
        commit_requested=True, production_activation_requested=True,
    )
    result = ex.QSpinActiveDryRunBridgeExecutor().execute(req)
    reasons = set(result.decision.block_reasons)
    assert ex.QSpinActiveDryRunBlockReason.FEATURE_GATE_MISSING in reasons
    assert ex.QSpinActiveDryRunBlockReason.SOURCE_MATRIX_MISSING in reasons
    assert ex.QSpinActiveDryRunBlockReason.PRODUCTION_ACTIVATION_REQUESTED in reasons

def test_active_dry_run_result_cannot_claim_live_effects():
    result = ex.QSpinActiveDryRunBridgeExecutor().execute(good_req("unsafe_result"))
    try:
        ex.QSpinActiveDryRunExecutionResult(result.request, result.decision, result.trace, result.audit_event, production_activated=True).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("live effect claim should fail")
