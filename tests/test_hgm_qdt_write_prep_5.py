from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrep2Options,
    HGMQDTWritePrep3Options,
    HGMQDTWritePrep4Options,
    HGMQDTWritePrep5Options,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    audit_permissioned_commit_boundary,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_3,
    build_hgm_qdt_write_prep_4,
    build_hgm_qdt_write_prep_5,
    build_hgm_qdt_write_prep_contracts,
    build_live_shape_contract_harness,
    build_production_write_blocker_burndown,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep5",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep5",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep5",
        source_type="ProceduralActionSequence",
        source_id="seq_prep5",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep5 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep5",
        trace_id="trace_payload_prep5",
        metadata={"secret_token": "do-not-leak-prep5"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep5",
        source_record_id="payload_prep5",
        target_slot_id="hgm_slot_prep5",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep5",
        dry_run=True,
        write_intent=False,
        confidence=0.93,
        trace_id="trace_hook_prep5",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep5",
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
    prep4 = build_hgm_qdt_write_prep_4(prep2, prep1_result=prep1, prep3_result=prep3, options=HGMQDTWritePrep4Options(max_tensor_dim=16))
    return prep1, prep2, prep3, prep4


def test_live_shape_contract_harness_accepts_valid_contract_preview():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    harness = build_live_shape_contract_harness(prep4, options=HGMQDTWritePrep5Options())
    assert harness.checks
    assert harness.live_shape_ready is True
    check = harness.checks[0]
    assert check.constructed is True
    assert check.content_rank_ok is True
    assert check.shape_matches is True
    assert check.write_permission_false is True
    assert check.live_shape_ready is True
    assert harness.metadata["system_commitgate_stage_called"] is False


def test_live_shape_contract_harness_invalid_input_fails_closed():
    harness = build_live_shape_contract_harness(object())
    assert harness.validation.ok is False
    assert not harness.checks
    assert harness.live_shape_ready is False


def test_permission_boundary_audit_keeps_stage_and_commit_blocked():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    harness = build_live_shape_contract_harness(prep4)
    audit = audit_permissioned_commit_boundary(prep4, harness)
    assert audit.checks
    assert audit.permission_boundary_clean is True
    assert audit.stage_called is False
    assert audit.commit_called is False
    assert audit.shared_slot_store_mutated is False
    assert audit.qh_storage_mutated is False
    assert audit.rollback_stack_mutated is False
    assert all(check.stage_blocked for check in audit.checks)
    assert all(check.commit_blocked for check in audit.checks)


def test_permission_boundary_invalid_input_fails_closed():
    audit = audit_permissioned_commit_boundary(object())
    assert audit.validation.ok is False
    assert not audit.checks
    assert audit.permission_boundary_clean is False


def test_blocker_burndown_records_open_production_blockers():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    harness = build_live_shape_contract_harness(prep4)
    audit = audit_permissioned_commit_boundary(prep4, harness)
    blockers = build_production_write_blocker_burndown(harness, audit, prep4)
    assert blockers.blockers
    assert blockers.production_write_ready is False
    assert blockers.open_blocker_count >= 1
    assert blockers.high_severity_open_count >= 1
    assert any(b.blocker_code == "ROLLBACK_SNAPSHOT_ACTUAL_BINDING" for b in blockers.blockers)
    assert any(b.status == "resolved" for b in blockers.blockers)


def test_blocker_burndown_can_be_bounded():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    harness = build_live_shape_contract_harness(prep4)
    audit = audit_permissioned_commit_boundary(prep4, harness)
    blockers = build_production_write_blocker_burndown(harness, audit, prep4, options={"max_blockers": 2})
    assert len(blockers.blockers) == 2
    assert blockers.validation.warnings


def test_high_level_write_prep_5_result_is_non_mutating_and_not_production_ready():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    result = build_hgm_qdt_write_prep_5(prep4)
    assert result.live_shape_harness.live_shape_ready is True
    assert result.permission_boundary_audit.permission_boundary_clean is True
    assert result.blocker_burndown.production_write_ready is False
    assert result.metadata["live_write_executed"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["system_commitgate_commit_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False
    assert result.metadata["qh_storage_mutated"] is False
    assert result.metadata["rollback_stack_mutated"] is False


def test_high_level_write_prep_5_invalid_input_fails_closed():
    result = build_hgm_qdt_write_prep_5(object())
    assert result.validation.ok is False
    assert result.live_shape_harness.live_shape_ready is False
    assert result.permission_boundary_audit.permission_boundary_clean is False
    assert result.blocker_burndown.production_write_ready is False


def test_trace_redaction_does_not_leak_secret_metadata():
    _prep1, _prep2, _prep3, prep4 = _prep_chain()
    result = build_hgm_qdt_write_prep_5(prep4)
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak-prep5" not in str(trace.redacted_payload())
