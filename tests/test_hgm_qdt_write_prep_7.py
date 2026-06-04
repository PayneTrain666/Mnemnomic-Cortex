from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrep2Options,
    HGMQDTWritePrep3Options,
    HGMQDTWritePrep4Options,
    HGMQDTWritePrep5Options,
    HGMQDTWritePrep6Options,
    HGMQDTWritePrep7Options,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_final_production_write_readiness_review,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_3,
    build_hgm_qdt_write_prep_4,
    build_hgm_qdt_write_prep_5,
    build_hgm_qdt_write_prep_6,
    build_hgm_qdt_write_prep_7,
    build_hgm_qdt_write_prep_contracts,
    build_permission_token_contract,
    build_shadow_commit_sandbox,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep7",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep7",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep7",
        source_type="ProceduralActionSequence",
        source_id="seq_prep7",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep7 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep7",
        trace_id="trace_payload_prep7",
        metadata={"secret_token": "do-not-leak-prep7"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep7",
        source_record_id="payload_prep7",
        target_slot_id="hgm_slot_prep7",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep7",
        dry_run=True,
        write_intent=False,
        confidence=0.95,
        trace_id="trace_hook_prep7",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep7",
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
    prep5 = build_hgm_qdt_write_prep_5(prep4, options=HGMQDTWritePrep5Options())
    prep6 = build_hgm_qdt_write_prep_6(prep4, prep5, options=HGMQDTWritePrep6Options(max_tensor_dim=16))
    return prep1, prep2, prep3, prep4, prep5, prep6


def test_permission_token_contract_preserves_denied_live_write_status():
    *_rest, prep6 = _prep_chain()
    contract = build_permission_token_contract(prep6)
    assert contract.token_records
    assert contract.token_contract_ready is False
    assert contract.live_write_authorized is False
    rec = contract.token_records[0]
    assert rec.permission_token_id_preview.startswith("ptok-")
    assert rec.token_present is False
    assert rec.write_permission_granted is False
    assert rec.can_authorize_live_write is False
    assert rec.blockers


def test_permission_token_invalid_input_fails_closed():
    contract = build_permission_token_contract(object())
    assert contract.validation.ok is False
    assert not contract.token_records
    assert contract.live_write_authorized is False


def test_shadow_commit_sandbox_simulates_without_live_mutation():
    *_rest, prep6 = _prep_chain()
    contract = build_permission_token_contract(prep6)
    shadow = build_shadow_commit_sandbox(prep6, contract)
    assert shadow.operations
    assert shadow.shadow_success is True
    assert shadow.live_stage_called is False
    assert shadow.live_commit_called is False
    assert shadow.live_store_mutated is False
    assert shadow.live_qh_mutated is False
    assert shadow.rollback_stack_mutated is False
    op = shadow.operations[0]
    assert op.shadow_stage_simulated is True
    assert op.shadow_commit_simulated is True
    assert op.live_stage_called is False
    assert op.live_commit_called is False


def test_shadow_commit_invalid_input_fails_closed():
    shadow = build_shadow_commit_sandbox(object())
    assert shadow.validation.ok is False
    assert not shadow.operations
    assert shadow.shadow_success is False
    assert shadow.live_commit_called is False


def test_final_blocker_review_preserves_open_production_blockers():
    *_rest, prep5, prep6 = _prep_chain()
    contract = build_permission_token_contract(prep6)
    shadow = build_shadow_commit_sandbox(prep6, contract)
    review = build_final_production_write_readiness_review(contract, shadow, prep6_result=prep6, prep5_result=prep5)
    assert review.blockers
    assert review.shadow_commit_ready is True
    assert review.permission_token_ready is False
    assert review.production_write_ready is False
    assert review.open_blocker_count >= 1
    assert review.high_severity_open_count >= 1
    assert any(b.blocker_code == "PRODUCTION_WRITE_PERMISSION_STAGE" for b in review.blockers)


def test_final_blocker_review_can_be_bounded():
    *_rest, prep5, prep6 = _prep_chain()
    contract = build_permission_token_contract(prep6)
    shadow = build_shadow_commit_sandbox(prep6, contract)
    review = build_final_production_write_readiness_review(contract, shadow, prep6_result=prep6, prep5_result=prep5, options={"max_final_blockers": 2})
    assert len(review.blockers) == 2
    assert review.validation.warnings


def test_high_level_write_prep_7_is_non_mutating_and_not_production_ready():
    *_rest, prep5, prep6 = _prep_chain()
    result = build_hgm_qdt_write_prep_7(prep6, prep5, options=HGMQDTWritePrep7Options())
    assert result.permission_token_contract.live_write_authorized is False
    assert result.shadow_commit_sandbox.shadow_success is True
    assert result.final_readiness_review.production_write_ready is False
    assert result.metadata["live_write_executed"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["system_commitgate_commit_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False
    assert result.metadata["qh_storage_mutated"] is False
    assert result.metadata["rollback_stack_mutated"] is False


def test_high_level_write_prep_7_invalid_input_fails_closed():
    result = build_hgm_qdt_write_prep_7(object())
    assert result.validation.ok is False
    assert not result.permission_token_contract.token_records
    assert not result.shadow_commit_sandbox.operations
    assert result.final_readiness_review.production_write_ready is False


def test_trace_redaction_does_not_leak_secret_metadata():
    *_rest, prep5, prep6 = _prep_chain()
    result = build_hgm_qdt_write_prep_7(prep6, prep5)
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak-prep7" not in str(trace.redacted_payload())
