from mnemonic_cortex.hypergraph_manifold import (
    HGM6CommitOptions,
    QDTWMBridgeOptions,
    build_hgm5_embedding_evaluation,
    build_hgm6_write_permission_gate,
    build_rollback_manifest,
    build_trace_safe_memory_plan,
    build_transaction_commit_preview,
    build_transaction_operation_previews,
    build_write_permission_state,
    score_commit_readiness,
)
from mnemonic_cortex.hypergraph_manifold.hgm3_result import ActionPrimitive, ProceduralActionSequence
from mnemonic_cortex.hypergraph_manifold.hgm6_result import RollbackManifest
from mnemonic_cortex.hypergraph_manifold.validation import ValidationResult


def make_sequence(sequence_id="seq_hgm6"):
    prim = ActionPrimitive(
        primitive_id="prim_hgm6_1",
        action_type="move",
        parameters={"dx": 1.0, "api_key": "must_redact"},
        duration=1.0,
        confidence=0.8,
    )
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=(prim,),
        source_hyperedge_id="he_hgm6_1",
        source_assignment_id="assign_hgm6_1",
        source_depth_target_id="target_hgm6_1",
        goal_label="reach_object",
        confidence=0.8,
        trace_id="trace_seq_hgm6",
        metadata={"password": "hide"},
    )


def make_plan(records=None, adapter_available=True, max_hook_count=128):
    records = records if records is not None else [make_sequence()]
    if adapter_available:
        opts = QDTWMBridgeOptions(max_hook_count=max_hook_count)
    else:
        opts = QDTWMBridgeOptions(
            expected_module_paths=("missing.hgm6.module",),
            expected_filesystem_paths=("missing/hgm6/path",),
            max_hook_count=max_hook_count,
        )
    return build_trace_safe_memory_plan(records, options=opts)


def test_write_permission_defaults_to_denied():
    perm = build_write_permission_state()
    assert perm.requested is False
    assert perm.granted is False
    assert perm.dry_run is True
    assert perm.preview_only is True
    assert "denied" in perm.reason


def test_explicit_preview_permission_still_does_not_execute_writes():
    plan = make_plan()
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True),
    )
    assert preview.write_permission.granted is True
    assert preview.metadata["executed"] is False
    assert preview.metadata["preview_only"] is True
    assert all(op.metadata["executed"] is False for op in preview.operations)


def test_missing_memory_plan_blocks_commit_preview():
    preview = build_transaction_commit_preview(None)
    assert preview.allowed is False
    assert preview.validation.errors
    assert "invalid" in preview.blocked_reason or preview.commit_readiness.blockers


def test_adapter_unavailable_blocks_commit_preview():
    plan = make_plan(adapter_available=False)
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True),
    )
    assert plan.adapter_status.available is False
    assert preview.allowed is False
    assert any("adapter unavailable" in blocker for blocker in preview.commit_readiness.blockers)


def test_rollback_manifest_is_generated_for_operation_previews():
    plan = make_plan()
    ops, validation, traces = build_transaction_operation_previews(plan)
    manifest = build_rollback_manifest(ops)
    assert validation.ok
    assert len(ops) == len(plan.slot_hooks)
    assert len(manifest.operations) == len(ops)
    assert manifest.complete is True or not any(op.allowed for op in ops)


def test_missing_rollback_coverage_blocks_commit_readiness():
    plan = make_plan()
    hgm5 = build_hgm5_embedding_evaluation(plan)
    incomplete = RollbackManifest(
        manifest_id="rb_incomplete",
        operations=tuple(),
        complete=False,
        validation=ValidationResult(),
    )
    readiness = score_commit_readiness(
        plan,
        hgm5_result=hgm5,
        rollback_manifest=incomplete,
        write_permission=build_write_permission_state(requested=True, granted=True),
        options=HGM6CommitOptions(requested=True, granted=True),
    )
    assert readiness.ready is False
    assert any("rollback" in blocker for blocker in readiness.blockers)


def test_transaction_operation_ordering_is_deterministic():
    plan_a = make_plan([make_sequence("seq_b"), make_sequence("seq_a")])
    plan_b = make_plan([make_sequence("seq_b"), make_sequence("seq_a")])
    ops_a, _, _ = build_transaction_operation_previews(plan_a)
    ops_b, _, _ = build_transaction_operation_previews(plan_b)
    assert [op.operation_id for op in ops_a] == [op.operation_id for op in ops_b]


def test_bounded_operation_count_is_enforced():
    records = [make_sequence(f"seq_{i}") for i in range(5)]
    plan = make_plan(records, max_hook_count=5)
    ops, validation, traces = build_transaction_operation_previews(plan, options=HGM6CommitOptions(max_operations=2))
    assert len(ops) == 2
    assert validation.warnings


def test_preflight_scoring_uses_hgm5_integration_score_when_available():
    plan = make_plan()
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(plan, hgm5_result=hgm5)
    assert preview.commit_readiness.metadata["hgm5_score"] == hgm5.integration_scoring_result.aggregate_score
    assert 0.0 <= preview.commit_readiness.score <= 1.0


def test_conservative_fallback_is_used_when_hgm5_score_is_missing():
    plan = make_plan()
    preview = build_transaction_commit_preview(plan, hgm5_result=None)
    assert preview.commit_readiness.metadata["hgm5_score"] == HGM6CommitOptions().conservative_missing_hgm5_score
    assert any("HGM-5" in warning for warning in preview.commit_readiness.warnings)


def test_trace_records_are_generated_and_redacted():
    plan = make_plan()
    result = build_hgm6_write_permission_gate(plan, options={"requested": True, "granted": False})
    assert result.trace_records
    payload = result.trace_records[-1].redacted_payload()
    assert payload.get("executed") is False
    # Existing HGM traces remain redaction-compatible.
    for trace in result.trace_records:
        redacted = trace.redacted_payload()
        assert "must_redact" not in str(redacted)


def test_high_level_hgm6_result_preserves_preview_only_behavior():
    plan = make_plan()
    hgm5 = build_hgm5_embedding_evaluation(plan)
    result = build_hgm6_write_permission_gate(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True),
    )
    assert result.metadata["executed"] is False
    assert result.transaction_preview.metadata["executed"] is False
    assert result.write_permission.preview_only is True
