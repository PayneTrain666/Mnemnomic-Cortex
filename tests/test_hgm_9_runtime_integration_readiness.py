from mnemonic_cortex.hypergraph_manifold import (
    HGM6CommitOptions,
    HGM8RuntimeOptions,
    HGM9ReadinessOptions,
    build_hgm5_embedding_evaluation,
    build_hgm6_write_permission_gate,
    build_hgm7_write_execution_adapter,
    build_hgm8_pipeline_evaluation,
    build_hgm9_runtime_integration_evaluation,
    build_trace_safe_memory_plan,
    benchmark_slot_lattice_replay,
    evaluate_qdt_runtime_integration,
    score_production_readiness,
)
from mnemonic_cortex.hypergraph_manifold.hgm3_result import ActionPrimitive, ProceduralActionSequence
from mnemonic_cortex.hypergraph_manifold.hgm4_result import SharedSlotLatticeHook


def make_sequence(sequence_id="seq_hgm9"):
    primitive = ActionPrimitive(
        primitive_id="prim_hgm9",
        action_type="move",
        parameters={"dx": 1.0, "secret_token": "must_redact"},
        duration=1.0,
        confidence=0.85,
    )
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=(primitive,),
        source_hyperedge_id="he_hgm9",
        source_assignment_id="assign_hgm9",
        source_depth_target_id="depth_hgm9",
        goal_label="reach_object",
        confidence=0.85,
        trace_id="trace_seq_hgm9",
        metadata={"password": "hide"},
    )


def make_hgm9_inputs(record_count=2):
    seqs = [make_sequence(f"seq_hgm9_{i}") for i in range(record_count)]
    plan = build_trace_safe_memory_plan(seqs)
    hgm5 = build_hgm5_embedding_evaluation(plan)
    hgm6 = build_hgm6_write_permission_gate(
        plan,
        hgm5_result=hgm5,
        options=HGM6CommitOptions(requested=True, granted=True, allow_commit_preview=True, readiness_threshold=0.1),
    )
    hgm7 = build_hgm7_write_execution_adapter(hgm6.transaction_preview)
    hgm8 = build_hgm8_pipeline_evaluation([hgm7, plan])
    return plan, hgm8, hgm7


def test_qdt_runtime_evaluation_scores_hgm8_result():
    plan, hgm8, _ = make_hgm9_inputs()
    result = evaluate_qdt_runtime_integration([hgm8, plan])
    assert result.validation.ok or result.validation.warnings
    assert result.metrics
    assert 0.0 <= result.integration_score <= 1.0
    assert result.metadata["live_qdt_write"] is False


def test_slot_lattice_replay_benchmark_scores_hooks():
    plan, _, _ = make_hgm9_inputs()
    result = benchmark_slot_lattice_replay(plan)
    assert result.validation.ok or result.validation.warnings
    assert result.records
    assert 0.0 <= result.aggregate_score <= 1.0
    assert all(record.metadata["live_qdt_write"] is False for record in result.records)


def test_slot_lattice_empty_input_returns_warning():
    result = benchmark_slot_lattice_replay([])
    assert result.validation.ok
    assert result.validation.warnings
    assert result.records == tuple()
    assert result.replay_safe is False


def test_slot_lattice_bounded_hook_count_is_enforced():
    plan, _, _ = make_hgm9_inputs(record_count=4)
    result = benchmark_slot_lattice_replay(plan, options=HGM9ReadinessOptions(max_replay_hooks=2))
    assert len(result.records) == 2
    assert result.validation.warnings


def test_production_readiness_gate_never_enables_production():
    plan, hgm8, _ = make_hgm9_inputs()
    qdt = evaluate_qdt_runtime_integration([hgm8, plan])
    replay = benchmark_slot_lattice_replay(plan)
    gate = score_production_readiness([hgm8, plan], qdt_evaluation=qdt, slot_replay=replay)
    assert 0.0 <= gate.score <= 1.0
    assert gate.production_enabled is False
    assert "production enablement" in " ".join(gate.blockers)


def test_high_level_hgm9_evaluation_builds_all_results():
    plan, hgm8, _ = make_hgm9_inputs()
    result = build_hgm9_runtime_integration_evaluation([hgm8, plan])
    assert result.validation.ok or result.validation.warnings
    assert result.qdt_runtime_evaluation.metrics
    assert result.slot_lattice_replay_benchmark.records
    assert 0.0 <= result.production_readiness_gate.score <= 1.0
    assert result.metadata["production_enabled"] is False


def test_missing_hgm8_degrades_safely():
    plan, _, _ = make_hgm9_inputs()
    result = build_hgm9_runtime_integration_evaluation(plan)
    assert result.validation.ok or result.validation.warnings
    assert result.qdt_runtime_evaluation.validation.warnings
    assert result.production_readiness_gate.score >= 0.0


def test_deterministic_runtime_evaluation_is_stable():
    plan, hgm8, _ = make_hgm9_inputs()
    a = build_hgm9_runtime_integration_evaluation([hgm8, plan], options=HGM9ReadinessOptions(readiness_threshold=0.1))
    b = build_hgm9_runtime_integration_evaluation([hgm8, plan], options=HGM9ReadinessOptions(readiness_threshold=0.1))
    assert a.production_readiness_gate.gate_id == b.production_readiness_gate.gate_id
    assert a.production_readiness_gate.score == b.production_readiness_gate.score


def test_trace_records_are_generated_and_redacted():
    plan, hgm8, _ = make_hgm9_inputs()
    result = build_hgm9_runtime_integration_evaluation([hgm8, plan])
    assert result.trace_records
    rendered = str([trace.redacted_payload() for trace in result.trace_records])
    assert "must_redact" not in rendered


def test_invalid_hook_replay_degrades_without_crash():
    plan, _, _ = make_hgm9_inputs()
    hook = plan.slot_hooks[0]
    bad = SharedSlotLatticeHook(
        hook_id=hook.hook_id + "_bad",
        source_record_id=hook.source_record_id,
        target_slot_id=hook.target_slot_id,
        depth_layer=hook.depth_layer,
        geometry_type=hook.geometry_type,
        qspin_signature_id=hook.qspin_signature_id,
        dry_run=False,
        write_intent=True,
        confidence=hook.confidence,
        trace_id="trace_bad_hook_hgm9",
    )
    result = benchmark_slot_lattice_replay([bad])
    assert result.records
    assert result.replay_safe is False
    assert result.validation.warnings


def test_require_adapter_available_can_block_runtime_ready():
    plan, hgm8, _ = make_hgm9_inputs()
    result = evaluate_qdt_runtime_integration([hgm8, plan], options=HGM9ReadinessOptions(require_adapter_available=True))
    if not result.adapter_available:
        assert result.runtime_ready is False


def test_hgm9_options_validate_positive_bounds():
    try:
        HGM9ReadinessOptions(max_records=0)
    except ValueError as exc:
        assert "max_records" in str(exc)
    else:
        raise AssertionError("expected max_records validation failure")


def test_hgm9_does_not_require_live_writes():
    plan, hgm8, _ = make_hgm9_inputs()
    result = build_hgm9_runtime_integration_evaluation([hgm8, plan])
    assert result.metadata["live_qdt_write"] is False
    assert result.production_readiness_gate.metadata["production_execution_enabled"] is False
