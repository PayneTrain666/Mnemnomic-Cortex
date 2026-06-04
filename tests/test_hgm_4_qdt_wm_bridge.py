import math

from mnemonic_cortex.hypergraph_manifold import (
    DepthLayer,
    GeometryType,
    QDTWMBridgeOptions,
    BridgePayloadBuildResult,
    HGMBridgePayload,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    build_bridge_payload_from_hgm_record,
    build_hgm4_qdt_wm_bridge,
    build_shared_slot_lattice_hooks,
    build_trace_safe_memory_plan,
    detect_qdt_wm_adapter_status,
    preview_bridge_execution,
)
from mnemonic_cortex.hypergraph_manifold.hgm1_result import BoundScenarioHyperedge
from mnemonic_cortex.hypergraph_manifold.hgm2_result import DepthRetrievalTarget, ManifoldRouteAssignment
from mnemonic_cortex.hypergraph_manifold.hgm3_result import (
    ActionPrimitive,
    ProceduralActionSequence,
    RoboticsPlanningActionOption,
    SPCPProcedureEmbedding,
)


def make_sequence(sequence_id="seq_demo"):
    prim = ActionPrimitive(
        primitive_id="prim_1",
        action_type="move",
        parameters={"dx": 1.0, "secret_token": "must_redact"},
        duration=1.0,
        confidence=0.8,
        metadata={"api_key": "hidden"},
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
        metadata={"password": "hide", "safe": "ok"},
    )


def make_embedding(sequence_id="seq_demo"):
    return SPCPProcedureEmbedding(
        embedding_id="emb_1",
        sequence_id=sequence_id,
        spherical_state=(1.0, 0.0),
        projective_state=(1.0, 0.0, 0.0, 1.0),
        conformal_warp=(0.01, 0.02),
        similarity_ready=True,
        trace_id="trace_emb",
        metadata={"credential": "hide"},
    )


def make_plan_option(sequence_id="seq_demo"):
    seq = make_sequence(sequence_id)
    return RoboticsPlanningActionOption(
        option_id="opt_1",
        sequence_id=sequence_id,
        action_primitives=seq.primitives,
        expected_goal="reach_object",
        risk_score=0.2,
        confidence=0.7,
        explanation="advisory only",
        trace_id="trace_opt",
        metadata={"token": "hide"},
    )


def test_valid_hgm3_procedural_sequence_builds_bridge_payload():
    result = build_bridge_payload_from_hgm_record(make_sequence())
    assert result.validation.ok
    assert result.payload is not None
    assert result.payload.source_type == "ProceduralActionSequence"
    assert result.payload.depth_layer == DepthLayer.D5_PROCEDURAL
    assert result.payload.geometry_type == GeometryType.SPCP


def test_valid_spcp_embedding_builds_bridge_payload():
    result = build_bridge_payload_from_hgm_record(make_embedding())
    assert result.validation.ok
    assert result.payload is not None
    assert result.payload.source_type == "SPCPProcedureEmbedding"
    assert result.payload.source_id == "emb_1"


def test_valid_robotics_planning_option_builds_bridge_payload():
    result = build_bridge_payload_from_hgm_record(make_plan_option())
    assert result.validation.ok
    assert result.payload is not None
    assert result.payload.source_type == "RoboticsPlanningActionOption"
    assert "advisory" in result.payload.content_summary.lower()


def test_unsupported_record_returns_structured_skip_result():
    result = build_bridge_payload_from_hgm_record(object())
    assert result.validation.ok  # warning-only skip
    assert result.payload is None
    assert result.metadata["skipped"] is True
    assert result.validation.warnings


def test_adapter_unavailable_status_returns_warning_not_crash():
    opts = QDTWMBridgeOptions(expected_module_paths=("missing.mod.path",), expected_filesystem_paths=("missing/path",))
    status = detect_qdt_wm_adapter_status(options=opts)
    assert status.available is False
    assert "unavailable" in status.reason.lower()


def test_shared_slot_lattice_hooks_are_generated():
    payload = build_bridge_payload_from_hgm_record(make_sequence()).payload
    result = build_shared_slot_lattice_hooks([payload])
    assert result.validation.ok
    assert len(result.hooks) == 1
    assert result.hooks[0].dry_run is True
    assert result.hooks[0].target_slot_id.startswith("hgm_slot_")


def test_bounded_hook_count_is_enforced():
    payloads = [build_bridge_payload_from_hgm_record(make_sequence(f"seq_{i}")).payload for i in range(5)]
    result = build_shared_slot_lattice_hooks(payloads, options=QDTWMBridgeOptions(max_hook_count=2))
    assert len(result.hooks) == 2
    assert result.validation.warnings


def test_dry_run_defaults_to_true():
    plan = build_trace_safe_memory_plan([make_sequence()])
    assert plan.dry_run is True
    assert all(hook.dry_run for hook in plan.slot_hooks)


def test_write_intent_defaults_to_false():
    plan = build_trace_safe_memory_plan([make_sequence()])
    assert plan.write_intent is False
    assert all(not hook.write_intent for hook in plan.slot_hooks)


def test_write_intent_is_blocked_unless_explicitly_allowed():
    opts = QDTWMBridgeOptions(write_intent=True, allow_write_preview=False)
    plan = build_trace_safe_memory_plan([make_sequence()], options=opts)
    preview = preview_bridge_execution(plan, options=opts)
    assert preview.allowed is False
    assert "blocked" in preview.blocked_reason.lower()


def test_write_intent_allowed_only_as_preview_when_explicitly_allowed():
    opts = QDTWMBridgeOptions(write_intent=True, allow_write_preview=True)
    plan = build_trace_safe_memory_plan([make_sequence()], options=opts)
    preview = preview_bridge_execution(plan, options=opts)
    assert preview.allowed is True
    assert preview.metadata["executed"] is False


def test_qspin_placeholder_is_generated_when_missing():
    payload = build_bridge_payload_from_hgm_record(make_sequence()).payload
    assert payload.qspin_signature_id.startswith("qspin_placeholder_")
    assert payload.metadata["qspin_placeholder_generated"] is True


def test_depth_layer_mapping_is_deterministic():
    payload = build_bridge_payload_from_hgm_record(make_sequence()).payload
    assert payload.depth_layer == DepthLayer.D5_PROCEDURAL
    hook1 = build_shared_slot_lattice_hooks([payload]).hooks[0]
    hook2 = build_shared_slot_lattice_hooks([payload]).hooks[0]
    assert hook1.target_slot_id == hook2.target_slot_id
    assert hook1.depth_layer == DepthLayer.D5_PROCEDURAL


def test_trace_payload_redaction_and_payload_metadata_redaction_work():
    result = build_bridge_payload_from_hgm_record(make_sequence())
    assert result.payload.metadata["password"] == "<redacted>"
    redacted = result.trace_records[0].redacted_payload()
    assert "source_id" in redacted


def test_bridge_plan_ordering_is_deterministic():
    records = [make_sequence("seq_b"), make_embedding("seq_a"), make_plan_option("seq_a")]
    plan1 = build_trace_safe_memory_plan(records)
    plan2 = build_trace_safe_memory_plan(records)
    assert [p.payload_id for p in plan1.bridge_payloads] == [p.payload_id for p in plan2.bridge_payloads]
    assert [h.target_slot_id for h in plan1.slot_hooks] == [h.target_slot_id for h in plan2.slot_hooks]


def test_missing_metadata_degrades_with_qspin_placeholder_and_warnings():
    result = build_bridge_payload_from_hgm_record(make_sequence())
    assert result.validation.ok
    assert result.payload.qspin_signature_id.startswith("qspin_placeholder_")


def test_hgm1_and_hgm2_records_convert_into_payloads():
    edge = BoundScenarioHyperedge(
        hyperedge_id="he_1",
        candidate_ids=("c1", "c2"),
        node_ids=("n1", "n2"),
        kind="scenario",
        coherence_score=0.8,
        probability_score=0.7,
        conflict_score=0.0,
        opportunity_score=0.1,
        source_candidate_indices=((0, 1), (0, 2)),
        trace_id="trace_he",
    )
    assignment = ManifoldRouteAssignment(
        assignment_id="assign_1",
        hyperedge_id="he_1",
        chart_id="chart_1",
        geometry_type=GeometryType.EUCLIDEAN,
        depth_layer=DepthLayer.D3_RELATION,
        distance_score=0.0,
        similarity_score=1.0,
        confidence=0.9,
        trace_id="trace_assign",
    )
    target = DepthRetrievalTarget(
        target_id="target_1",
        hyperedge_id="he_1",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        retrieval_key="key_1",
        priority=0.8,
        trace_id="trace_target",
    )
    payloads = [build_bridge_payload_from_hgm_record(item).payload for item in (edge, assignment, target)]
    assert [p.source_type for p in payloads] == ["BoundScenarioHyperedge", "ManifoldRouteAssignment", "DepthRetrievalTarget"]


def test_build_hgm4_qdt_wm_bridge_end_to_end():
    result = build_hgm4_qdt_wm_bridge([make_sequence(), make_embedding(), make_plan_option()])
    assert result.validation.ok or result.validation.warnings
    assert isinstance(result.memory_plan, TraceSafeMemoryPlan)
    assert len(result.memory_plan.bridge_payloads) == 3
    assert result.execution_preview.metadata["executed"] is False
