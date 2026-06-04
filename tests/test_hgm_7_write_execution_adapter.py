from mnemonic_cortex.hypergraph_manifold import (
    HGM6CommitOptions,
    HGM7ExecutionOptions,
    QDTWMBridgeOptions,
    build_hgm5_embedding_evaluation,
    build_hgm7_write_execution_adapter,
    build_trace_safe_memory_plan,
    build_transaction_commit_preview,
    build_transaction_log,
    build_write_execution_adapter_status,
    execute_write_adapter,
    verify_recovery,
)
from mnemonic_cortex.hypergraph_manifold.hgm3_result import ActionPrimitive, ProceduralActionSequence
from mnemonic_cortex.hypergraph_manifold.hgm6_result import RollbackManifest
from mnemonic_cortex.hypergraph_manifold.validation import ValidationResult


def make_sequence(sequence_id="seq_hgm7"):
    prim = ActionPrimitive(
        primitive_id=f"prim_{sequence_id}",
        action_type="move",
        parameters={"dx": 1.0, "api_key": "must_redact"},
        duration=1.0,
        confidence=0.8,
    )
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=(prim,),
        source_hyperedge_id=f"he_{sequence_id}",
        source_assignment_id=f"assign_{sequence_id}",
        source_depth_target_id=f"target_{sequence_id}",
        goal_label="reach_object",
        confidence=0.8,
        trace_id=f"trace_{sequence_id}",
        metadata={"password": "hide"},
    )


def make_plan(records=None, adapter_available=True, max_hook_count=128):
    records = records if records is not None else [make_sequence()]
    if adapter_available:
        opts = QDTWMBridgeOptions(max_hook_count=max_hook_count)
    else:
        opts = QDTWMBridgeOptions(
            expected_module_paths=("missing.hgm7.module",),
            expected_filesystem_paths=("missing/hgm7/path",),
            max_hook_count=max_hook_count,
        )
    return build_trace_safe_memory_plan(records, options=opts)


def make_allowed_preview(record_count=1):
    records = [make_sequence(f"seq_hgm7_{i}") for i in range(record_count)]
    plan = make_plan(records=records, max_hook_count=max(record_count, 1))
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(
            requested=True,
            granted=True,
            allow_commit_preview=True,
            readiness_threshold=0.1,
            max_operations=max(record_count, 1),
        ),
    )
    assert preview.operations
    return preview


def test_default_simulation_mode_logs_without_real_execution():
    preview = make_allowed_preview()
    result = execute_write_adapter(preview)
    assert result.simulation_mode is True
    assert result.executed is False
    assert result.metadata["live_qdt_write"] is False
    assert all(entry.status in {"simulated", "blocked"} for entry in result.transaction_log.entries)


def test_invalid_preview_is_blocked_safely():
    result = execute_write_adapter(None)
    assert result.allowed is False
    assert result.executed is False
    assert result.validation.errors
    assert "invalid" in result.blocked_reason


def test_transaction_log_ordering_is_deterministic():
    preview_a = make_allowed_preview(record_count=3)
    preview_b = make_allowed_preview(record_count=3)
    log_a = build_transaction_log(preview_a)
    log_b = build_transaction_log(preview_b)
    assert [entry.operation_id for entry in log_a.entries] == [entry.operation_id for entry in log_b.entries]


def test_bounded_transaction_log_count_is_enforced():
    preview = make_allowed_preview(record_count=5)
    log = build_transaction_log(preview, options=HGM7ExecutionOptions(max_operations=2))
    assert len(log.entries) == 2
    assert log.validation.warnings


def test_non_simulation_without_test_enable_is_unavailable():
    status = build_write_execution_adapter_status(options=HGM7ExecutionOptions(simulation_mode=False, allow_test_execution=False))
    assert status.available is False
    assert "without allow_test_execution" in status.reason


def test_test_execution_path_is_isolated_and_explicit():
    preview = make_allowed_preview()
    result = execute_write_adapter(preview, options=HGM7ExecutionOptions(simulation_mode=False, allow_test_execution=True))
    assert result.executed is True
    assert result.metadata["live_qdt_write"] is False
    assert all(entry.status == "test_executed" for entry in result.transaction_log.entries)


def test_recovery_verification_passes_with_complete_manifest():
    preview = make_allowed_preview()
    log = build_transaction_log(preview)
    recovery = verify_recovery(preview.rollback_manifest, log)
    assert recovery.rollback_ready is True
    assert all(record.verified for record in recovery.records)


def test_recovery_verification_fails_with_missing_manifest_coverage():
    preview = make_allowed_preview()
    log = build_transaction_log(preview)
    incomplete = RollbackManifest(
        manifest_id="rb_missing_hgm7",
        operations=tuple(),
        complete=False,
        validation=ValidationResult(),
    )
    recovery = verify_recovery(incomplete, log)
    assert recovery.rollback_ready is False
    assert recovery.validation.errors


def test_commit_preview_not_allowed_blocks_execution_adapter():
    plan = make_plan(adapter_available=False)
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True, readiness_threshold=0.1),
    )
    result = execute_write_adapter(preview)
    assert result.allowed is False
    assert result.executed is False
    assert result.blocked_reason


def test_high_level_hgm7_result_preserves_preview_safety():
    preview = make_allowed_preview()
    result = build_hgm7_write_execution_adapter(preview)
    assert result.execution_result.executed is False
    assert result.metadata["live_qdt_write"] is False
    assert result.trace_records


def test_trace_records_are_generated_and_redacted():
    preview = make_allowed_preview()
    result = build_hgm7_write_execution_adapter(preview)
    assert result.trace_records
    for trace in result.trace_records:
        redacted = trace.redacted_payload()
        assert "must_redact" not in str(redacted)
