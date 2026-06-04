from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
act = mods["qspin_runtime_shadow_activation"]
commit = mods["qspin_commit_gate_dry_run"]
conf = mods["qspin_production_config"]
ks = mods["qspin_kill_switch"]
rb = mods["qspin_rollback_harness"]


def safe_inputs():
    kill = ks.build_default_qspin_kill_switch().decision()
    cg = commit.QSpinCommitGateDryRunInspector().inspect(commit.QSpinCommitGateDryRunRequest("cg", source_matrix_complete=True, rollback_evidence_present=True))
    evidence = tuple(rb.QSpinRollbackEvidenceRecord("ev_" + kind.value, kind) for kind in rb.QSpinRollbackEvidenceKind)
    rollback = rb.build_default_qspin_rollback_harness().dry_run(rb.QSpinRollbackDryRunRequest("rb", evidence))
    return kill, cg, rollback


def test_default_config_is_safe():
    cfg = act.build_default_qspin_shadow_activation_config()
    assert cfg.allow_active_routing is False
    assert cfg.allow_payload_transfer is False
    assert cfg.allow_writes is False
    assert cfg.allow_commit_execution is False


def test_shadow_activation_allowed_only_when_all_gates_pass():
    kill, cg, rollback = safe_inputs()
    controller = act.QSpinRuntimeShadowActivationController()
    req = act.QSpinRuntimeShadowActivationRequest("shadow_ok", source_matrix_complete=True)
    result = controller.evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert result.decision.allowed_shadow_only is True
    assert result.routed_live_data is False
    assert result.transferred_payload is False
    assert result.wrote_state is False
    assert result.executed_commit is False


def test_idempotent_repeated_request():
    kill, cg, rollback = safe_inputs()
    controller = act.QSpinRuntimeShadowActivationController()
    req = act.QSpinRuntimeShadowActivationRequest("same", source_matrix_complete=True)
    first = controller.evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    second = controller.evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert first.decision.status is act.QSpinShadowActivationStatus.ALLOWED_SHADOW_ONLY
    assert second.decision.status is act.QSpinShadowActivationStatus.IDEMPOTENT_REPLAY


def test_unsafe_feature_flags_block():
    kill, cg, rollback = safe_inputs()
    req = act.QSpinRuntimeShadowActivationRequest("bad", source_matrix_complete=True, feature_flags=act.QSpinRuntimeFeatureFlagSnapshot(frozenset({conf.QSpinRuntimeFeatureFlag.PAYLOAD_TRANSFER})))
    result = act.QSpinRuntimeShadowActivationController().evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert act.QSpinShadowActivationBlockReason.PAYLOAD_TRANSFER_REQUESTED in result.decision.block_reasons


def test_missing_matrix_kill_switch_commit_gate_or_rollback_blocks():
    kill, cg, rollback = safe_inputs()
    controller = act.QSpinRuntimeShadowActivationController()
    req = act.QSpinRuntimeShadowActivationRequest("no_matrix", source_matrix_complete=False)
    result = controller.evaluate(req, kill_switch_decision=kill, commit_gate_result=cg, rollback_result=rollback)
    assert act.QSpinShadowActivationBlockReason.SOURCE_MATRIX_INCOMPLETE in result.decision.block_reasons
