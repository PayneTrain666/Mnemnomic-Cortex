from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrep2Options,
    HGMQDTWritePrep3Options,
    HGMQDTWritePrep4Options,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_3,
    build_hgm_qdt_write_prep_4,
    build_hgm_qdt_write_prep_contracts,
    build_real_contract_object_previews,
    build_rollback_snapshot_binding_plan,
    build_synthetic_commitgate_adapter_boundary,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep4",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep4",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep4",
        source_type="ProceduralActionSequence",
        source_id="seq_prep4",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep4 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep4",
        trace_id="trace_payload_prep4",
        metadata={"secret_token": "do-not-leak"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep4",
        source_record_id="payload_prep4",
        target_slot_id="hgm_slot_prep4",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep4",
        dry_run=True,
        write_intent=False,
        confidence=0.91,
        trace_id="trace_hook_prep4",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep4",
        bridge_payloads=(payload,),
        slot_hooks=(hook,),
        adapter_status=adapter,
        dry_run=True,
        write_intent=False,
        validation=ValidationResult(),
        trace_records=tuple(),
        metadata={},
    )


def _prep_chain():
    prep1 = build_hgm_qdt_write_prep_contracts(_sample_memory_plan(), options=HGMQDTWritePrepOptions(proposal_dim=8))
    prep2 = build_hgm_qdt_write_prep_2(prep1, options=HGMQDTWritePrep2Options(max_tensor_dim=16))
    prep3 = build_hgm_qdt_write_prep_3(prep2, options=HGMQDTWritePrep3Options())
    return prep1, prep2, prep3


def test_real_contract_object_preview_constructs_without_live_write():
    _prep1, prep2, _prep3 = _prep_chain()
    result = build_real_contract_object_previews(prep2, options=HGMQDTWritePrep4Options(max_tensor_dim=16))
    assert len(result.previews) == 1
    assert result.constructed_count == 1
    preview = result.previews[0]
    assert preview.constructed is True
    assert preview.contract_class_name == "SystemWriteProposal"
    assert preview.proposal_id.startswith("sysprop-")
    assert preview.write_permission is False
    assert preview.object_trace["write_permission"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False


def test_real_contract_object_construction_can_be_disabled():
    _prep1, prep2, _prep3 = _prep_chain()
    result = build_real_contract_object_previews(prep2, options={"allow_real_contract_construction": False})
    assert result.previews
    assert result.constructed_count == 0
    assert result.previews[0].constructed is False
    assert result.previews[0].blockers


def test_adapter_boundary_blocks_stage_and_commit():
    _prep1, prep2, _prep3 = _prep_chain()
    contracts = build_real_contract_object_previews(prep2)
    boundary = build_synthetic_commitgate_adapter_boundary(contracts)
    assert boundary.checks
    assert boundary.stage_called is False
    assert boundary.commit_called is False
    assert boundary.shared_slot_store_mutated is False
    assert boundary.qh_storage_mutated is False
    assert boundary.rollback_stack_mutated is False
    assert all(not check.stage_allowed for check in boundary.checks)
    assert all(not check.commit_allowed for check in boundary.checks)


def test_invalid_boundary_input_fails_closed():
    boundary = build_synthetic_commitgate_adapter_boundary(object())
    assert boundary.validation.ok is False
    assert not boundary.checks
    assert boundary.stage_called is False
    assert boundary.commit_called is False


def test_rollback_snapshot_binding_plan_remains_not_live_ready():
    prep1, _prep2, prep3 = _prep_chain()
    plan = build_rollback_snapshot_binding_plan(prep1.rollback_handshake, prep3)
    assert plan.bindings
    assert plan.ready_for_live_commit is False
    assert plan.complete is False
    assert all(binding.actual_rollback_stack_required for binding in plan.bindings)
    assert all(not binding.bound_to_actual_snapshot for binding in plan.bindings)
    assert all(binding.synthetic_snapshot_ref.startswith("synthetic_snapshot_") for binding in plan.bindings)
    assert plan.metadata["rollback_stack_mutated"] is False


def test_invalid_rollback_binding_input_fails_closed():
    plan = build_rollback_snapshot_binding_plan(object())
    assert plan.validation.ok is False
    assert not plan.bindings
    assert plan.ready_for_live_commit is False


def test_high_level_write_prep_4_result_is_non_mutating():
    prep1, prep2, prep3 = _prep_chain()
    result = build_hgm_qdt_write_prep_4(prep2, prep1_result=prep1, prep3_result=prep3)
    assert result.contract_object_result.previews
    assert result.adapter_boundary.checks
    assert result.rollback_binding_plan.bindings
    assert result.metadata["live_write_executed"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["system_commitgate_commit_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False
    assert result.metadata["qh_storage_mutated"] is False
    assert result.metadata["rollback_stack_mutated"] is False


def test_trace_redaction_does_not_leak_secret_metadata():
    prep1, prep2, prep3 = _prep_chain()
    result = build_hgm_qdt_write_prep_4(prep2, prep1_result=prep1, prep3_result=prep3)
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak" not in str(trace.redacted_payload())


def test_bounded_contract_object_count_is_enforced():
    _prep1, prep2, _prep3 = _prep_chain()
    proposals = prep2.proposal_builder_result.proposals * 4
    result = build_real_contract_object_previews(proposals, options={"max_contract_objects": 2})
    assert len(result.previews) == 2
    assert result.validation.warnings
