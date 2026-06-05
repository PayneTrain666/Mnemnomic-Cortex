from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
commit = mods["qspin_commit_gate_dry_run"]
conf = mods["qspin_production_config"]


def test_safe_dry_run_permits_shadow_only():
    inspector = commit.QSpinCommitGateDryRunInspector()
    req = commit.QSpinCommitGateDryRunRequest("safe", source_matrix_complete=True, rollback_evidence_present=True, kill_switch_state=conf.QSpinRuntimeKillSwitchState.ENABLED)
    result = inspector.inspect(req)
    assert result.decision.allowed_for_shadow is True
    assert result.commit_executed is False
    assert result.runtime_activated is False


def test_unsafe_runtime_flags_block():
    inspector = commit.QSpinCommitGateDryRunInspector()
    req = commit.QSpinCommitGateDryRunRequest("bad", requested_feature_flags=frozenset({conf.QSpinRuntimeFeatureFlag.BRIDGE_ROUTING}), source_matrix_complete=True, rollback_evidence_present=True)
    result = inspector.inspect(req)
    assert result.decision.allowed_for_shadow is False
    assert commit.QSpinCommitGateDryRunBlockReason.UNSAFE_RUNTIME_FLAG in result.decision.blocked_reasons


def test_missing_source_and_rollback_block():
    result = commit.QSpinCommitGateDryRunInspector().inspect(commit.QSpinCommitGateDryRunRequest("missing"))
    assert commit.QSpinCommitGateDryRunBlockReason.SOURCE_MATRIX_MISSING in result.decision.blocked_reasons
    assert commit.QSpinCommitGateDryRunBlockReason.ROLLBACK_EVIDENCE_MISSING in result.decision.blocked_reasons


def test_disabled_kill_switch_blocks():
    result = commit.QSpinCommitGateDryRunInspector().inspect(commit.QSpinCommitGateDryRunRequest("ks", source_matrix_complete=True, rollback_evidence_present=True, kill_switch_state=conf.QSpinRuntimeKillSwitchState.DISABLED))
    assert commit.QSpinCommitGateDryRunBlockReason.KILL_SWITCH_DISABLED in result.decision.blocked_reasons


def test_raw_trace_write_and_production_activation_block():
    req = commit.QSpinCommitGateDryRunRequest("unsafe", source_matrix_complete=True, rollback_evidence_present=True, write_permissions_requested=True, raw_payload_trace_requested=True, production_activation_requested=True, commit_execution_requested=True)
    result = commit.QSpinCommitGateDryRunInspector().inspect(req)
    reasons = set(result.decision.blocked_reasons)
    assert commit.QSpinCommitGateDryRunBlockReason.WRITE_PERMISSION_REQUESTED in reasons
    assert commit.QSpinCommitGateDryRunBlockReason.RAW_PAYLOAD_TRACE_REQUESTED in reasons
    assert commit.QSpinCommitGateDryRunBlockReason.PRODUCTION_ACTIVATION_REQUESTED in reasons
    assert commit.QSpinCommitGateDryRunBlockReason.COMMIT_EXECUTION_REQUESTED in reasons
