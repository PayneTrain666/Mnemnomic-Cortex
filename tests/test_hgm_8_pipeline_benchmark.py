from mnemonic_cortex.hypergraph_manifold import (
    HGM6CommitOptions,
    HGM7ExecutionOptions,
    HGM8RuntimeOptions,
    build_baseline_hgm_embeddings,
    build_hgm5_embedding_evaluation,
    build_hgm6_write_permission_gate,
    build_hgm7_write_execution_adapter,
    build_hgm8_pipeline_evaluation,
    build_runtime_embedding_trainer,
    benchmark_hgm_pipeline,
    evaluate_safe_write_replay,
    build_trace_safe_memory_plan,
    build_transaction_commit_preview,
)
from mnemonic_cortex.hypergraph_manifold.hgm3_result import ActionPrimitive, ProceduralActionSequence
from mnemonic_cortex.hypergraph_manifold.hgm7_result import TransactionLogEntry, TransactionLog
from mnemonic_cortex.hypergraph_manifold.validation import ValidationResult


def make_sequence(sequence_id="seq_hgm8"):
    primitive = ActionPrimitive(
        primitive_id="prim_hgm8",
        action_type="move",
        parameters={"dx": 1.0, "secret_token": "must_redact"},
        duration=1.0,
        confidence=0.85,
    )
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=(primitive,),
        source_hyperedge_id="he_hgm8",
        source_assignment_id="assign_hgm8",
        source_depth_target_id="depth_hgm8",
        goal_label="reach_object",
        confidence=0.85,
        trace_id="trace_seq_hgm8",
        metadata={"password": "hide"},
    )


def make_allowed_hgm7_result(record_count=2):
    seqs = [make_sequence(f"seq_hgm8_{i}") for i in range(record_count)]
    plan = build_trace_safe_memory_plan(seqs)
    hgm5 = build_hgm5_embedding_evaluation(plan)
    preview = build_transaction_commit_preview(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True, readiness_threshold=0.1),
    )
    return build_hgm7_write_execution_adapter(preview)


def test_runtime_embedding_trainer_accepts_hgm5_embeddings():
    baseline = build_baseline_hgm_embeddings([])
    seq = make_sequence()
    plan = build_trace_safe_memory_plan([seq])
    hgm5 = build_hgm5_embedding_evaluation(plan)
    result = build_runtime_embedding_trainer(hgm5.trainer_result.embeddings)
    assert result.validation.ok or result.validation.warnings
    assert result.embeddings
    assert len(result.embeddings[0].vector) == HGM8RuntimeOptions().embedding_dimension
    assert result.embeddings[0].learned_runtime_ready is False


def test_runtime_embedding_trainer_accepts_hgm7_result():
    hgm7 = make_allowed_hgm7_result()
    result = build_runtime_embedding_trainer([hgm7])
    assert result.validation.ok
    assert len(result.embeddings) == 1
    assert result.embeddings[0].source_type == "HGM7WriteExecutionResult"


def test_empty_runtime_embedding_input_returns_warning():
    result = build_runtime_embedding_trainer([])
    assert result.validation.ok
    assert result.validation.warnings
    assert result.embeddings == tuple()


def test_unsupported_embedding_records_degrade_safely():
    result = build_runtime_embedding_trainer([object()])
    assert result.validation.ok
    assert result.validation.warnings
    assert result.embeddings == tuple()


def test_deterministic_runtime_embedding_generation_is_stable():
    hgm7 = make_allowed_hgm7_result()
    opts = HGM8RuntimeOptions(deterministic_seed=44, embedding_dimension=10)
    a = build_runtime_embedding_trainer([hgm7], options=opts).embeddings[0]
    b = build_runtime_embedding_trainer([hgm7], options=opts).embeddings[0]
    assert a.embedding_id == b.embedding_id
    assert a.vector == b.vector
    assert len(a.vector) == 10


def test_safe_write_replay_evaluates_simulated_logs_without_writes():
    hgm7 = make_allowed_hgm7_result()
    result = evaluate_safe_write_replay(hgm7)
    assert result.validation.ok or result.validation.warnings
    assert result.records
    assert result.safe is True
    assert all(record.replayed is False for record in result.records)
    assert all(record.metadata["live_qdt_write"] is False for record in result.records)


def test_safe_write_replay_empty_input_returns_warning():
    result = evaluate_safe_write_replay(None)
    assert result.validation.ok
    assert result.validation.warnings
    assert result.records == tuple()
    assert result.safe is False


def test_safe_write_replay_blocks_unknown_status_fail_closed():
    entry = TransactionLogEntry(
        entry_id="entry_unknown_hgm8",
        operation_id="op_unknown_hgm8",
        operation_type="write_preview",
        status="mystery",
        source_payload_id="payload_x",
        target_slot_id="slot_x",
        dry_run=True,
        simulation_mode=True,
        trace_id="trace_unknown_hgm8",
    )
    result = evaluate_safe_write_replay(entry)
    assert result.validation.errors
    assert result.safe is False
    assert result.records[0].safe is False


def test_safe_write_replay_respects_bounded_log_entries():
    hgm7 = make_allowed_hgm7_result(record_count=5)
    result = evaluate_safe_write_replay(hgm7, options=HGM8RuntimeOptions(max_log_entries=2))
    assert len(result.records) == 2
    assert result.validation.warnings


def test_pipeline_benchmark_generates_metrics():
    hgm7 = make_allowed_hgm7_result()
    result = benchmark_hgm_pipeline(hgm7)
    assert result.validation.ok or result.validation.warnings
    assert result.metrics
    assert 0.0 <= result.aggregate_score <= 1.0
    assert any(metric.metric_name == "no_live_write_guarantee" for metric in result.metrics)


def test_high_level_hgm8_pipeline_evaluation():
    hgm7 = make_allowed_hgm7_result()
    result = build_hgm8_pipeline_evaluation(hgm7)
    assert result.validation.ok or result.validation.warnings
    assert result.trainer_result.embeddings
    assert result.replay_result.records
    assert 0.0 <= result.benchmark_result.aggregate_score <= 1.0
    assert result.metadata["live_qdt_write"] is False


def test_trace_records_are_generated_and_redacted():
    hgm7 = make_allowed_hgm7_result()
    result = build_hgm8_pipeline_evaluation(hgm7)
    assert result.trace_records
    assert "must_redact" not in str([trace.redacted_payload() for trace in result.trace_records])


def test_bounded_runtime_embedding_record_count_is_enforced():
    hgm7 = make_allowed_hgm7_result(record_count=4)
    entries = list(hgm7.execution_result.transaction_log.entries)
    result = build_runtime_embedding_trainer(entries, options=HGM8RuntimeOptions(max_records=2))
    assert len(result.embeddings) == 2
    assert result.validation.warnings
