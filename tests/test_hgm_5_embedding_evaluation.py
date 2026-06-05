from mnemonic_cortex.hypergraph_manifold import (
    DepthLayer,
    GeometryType,
    EmbeddingTrainerOptions,
    HGMBridgePayload,
    SharedSlotLatticeHook,
    build_baseline_hgm_embeddings,
    build_hgm5_embedding_evaluation,
    build_trace_safe_memory_plan,
    build_bridge_payload_from_hgm_record,
    evaluate_bridge_payload_quality,
    evaluate_execution_preview_quality,
    evaluate_slot_hook_quality,
    preview_bridge_execution,
    score_hgm_integration_readiness,
)
from mnemonic_cortex.hypergraph_manifold.hgm3_result import ActionPrimitive, ProceduralActionSequence


def make_payload(source_id="src_1"):
    return HGMBridgePayload(
        payload_id=f"payload_{source_id}",
        source_type="ProceduralActionSequence",
        source_id=source_id,
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="procedure summary",
        qspin_signature_id="qspin_1",
        trace_id="trace_payload",
        metadata={"api_key": "<redacted>", "safe": "ok"},
    )


def make_hook(source_id="src_1"):
    return SharedSlotLatticeHook(
        hook_id=f"hook_{source_id}",
        source_record_id=source_id,
        target_slot_id=f"hgm_slot_5_{source_id}",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_1",
        dry_run=True,
        write_intent=False,
        confidence=0.8,
        trace_id="trace_hook",
    )


def make_sequence(sequence_id="seq_demo"):
    prim = ActionPrimitive(
        primitive_id="prim_1",
        action_type="move",
        parameters={"dx": 1.0, "secret_token": "must_redact"},
        duration=1.0,
        confidence=0.8,
    )
    return ProceduralActionSequence(
        sequence_id=sequence_id,
        primitives=(prim,),
        source_hyperedge_id="he_1",
        source_assignment_id="assign_1",
        source_depth_target_id="target_1",
        goal_label="reach_object",
        confidence=0.8,
        trace_id="trace_seq",
        metadata={"password": "hide"},
    )


def test_valid_hgm_bridge_payload_builds_baseline_embedding():
    result = build_baseline_hgm_embeddings([make_payload()])
    assert result.validation.ok
    assert len(result.embeddings) == 1
    emb = result.embeddings[0]
    assert emb.source_id == "src_1"
    assert emb.source_type == "HGMBridgePayload"
    assert len(emb.vector) == EmbeddingTrainerOptions().embedding_dimension
    assert all(abs(v) <= 1.0 for v in emb.vector)


def test_valid_shared_slot_lattice_hook_builds_baseline_embedding():
    result = build_baseline_hgm_embeddings([make_hook()])
    assert result.validation.ok
    assert len(result.embeddings) == 1
    assert result.embeddings[0].source_type == "SharedSlotLatticeHook"


def test_empty_input_returns_structured_warning_result():
    result = build_baseline_hgm_embeddings([])
    assert result.validation.ok
    assert result.validation.warnings
    assert result.embeddings == tuple()
    top = build_hgm5_embedding_evaluation([])
    assert top.validation.ok
    assert top.validation.warnings


def test_unsupported_records_degrade_safely():
    result = build_baseline_hgm_embeddings([object()])
    assert result.validation.ok
    assert result.validation.warnings
    assert result.embeddings == tuple()


def test_deterministic_embedding_generation_is_stable():
    payload = make_payload()
    opts = EmbeddingTrainerOptions(deterministic_seed=99, embedding_dimension=12)
    a = build_baseline_hgm_embeddings([payload], options=opts).embeddings[0]
    b = build_baseline_hgm_embeddings([payload], options=opts).embeddings[0]
    assert a.embedding_id == b.embedding_id
    assert a.vector == b.vector
    assert len(a.vector) == 12


def test_payload_quality_metrics_are_generated():
    result = evaluate_bridge_payload_quality([make_payload()])
    assert result.validation.ok
    assert result.metrics
    assert 0.0 <= result.aggregate_score <= 1.0
    names = {m.metric_name for m in result.metrics}
    assert "source_id_completeness" in names
    assert "qspin_presence" in names


def test_slot_hook_quality_metrics_are_generated():
    result = evaluate_slot_hook_quality([make_hook()])
    assert result.validation.ok
    assert result.metrics
    assert 0.0 <= result.aggregate_score <= 1.0
    names = {m.metric_name for m in result.metrics}
    assert "dry_run_safety" in names
    assert "target_slot_id_stability" in names


def test_execution_preview_quality_metrics_are_generated():
    seq = make_sequence()
    payload = build_bridge_payload_from_hgm_record(seq).payload
    plan = build_trace_safe_memory_plan([seq])
    preview = preview_bridge_execution(plan)
    result = evaluate_execution_preview_quality(preview)
    assert payload is not None
    assert result.validation.ok
    assert result.metrics
    assert 0.0 <= result.aggregate_score <= 1.0
    assert any(m.metric_name == "no_execution_guarantee" for m in result.metrics)


def test_integration_readiness_score_is_finite_and_bounded():
    plan = build_trace_safe_memory_plan([make_sequence()])
    preview = preview_bridge_execution(plan)
    records = list(plan.bridge_payloads) + list(plan.slot_hooks) + [preview]
    result = score_hgm_integration_readiness(records)
    assert result.validation.ok or result.validation.warnings
    assert result.scores
    assert 0.0 <= result.aggregate_score <= 1.0


def test_bounded_record_count_is_enforced():
    records = [make_payload(f"src_{i}") for i in range(5)]
    result = build_baseline_hgm_embeddings(records, options=EmbeddingTrainerOptions(max_records=2))
    assert len(result.embeddings) == 2
    assert result.validation.warnings


def test_optional_dependency_fallback_does_not_crash():
    result = build_baseline_hgm_embeddings(
        [make_payload()],
        options=EmbeddingTrainerOptions(allow_optional_numpy=True, allow_optional_torch=True),
    )
    assert result.validation.ok
    assert "numpy_available" in result.metadata
    assert "torch_available" in result.metadata


def test_trace_records_are_generated_and_redacted():
    result = evaluate_bridge_payload_quality([make_payload()])
    assert result.trace_records
    redacted = result.trace_records[0].redacted_payload()
    assert "metadata" in redacted or "source_id" in redacted


def test_high_level_hgm5_evaluation_accepts_trace_safe_plan():
    plan = build_trace_safe_memory_plan([make_sequence()])
    result = build_hgm5_embedding_evaluation(plan)
    assert result.validation.ok or result.validation.warnings
    assert result.trainer_result.embeddings
    assert 0.0 <= result.integration_scoring_result.aggregate_score <= 1.0
    assert result.metadata["evaluation_first"] is True
