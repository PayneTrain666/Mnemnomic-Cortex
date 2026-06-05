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
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_3,
    build_hgm_qdt_write_prep_4,
    build_hgm_qdt_write_prep_5,
    build_hgm_qdt_write_prep_6,
    build_hgm_qdt_write_prep_contracts,
    build_qh_storage_record_sandbox,
    build_real_shared_slot_store_parity_harness,
    build_rollback_snapshot_binding_dry_run,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep6",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep6",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep6",
        source_type="ProceduralActionSequence",
        source_id="seq_prep6",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep6 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep6",
        trace_id="trace_payload_prep6",
        metadata={"secret_token": "do-not-leak-prep6"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep6",
        source_record_id="payload_prep6",
        target_slot_id="hgm_slot_prep6",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep6",
        dry_run=True,
        write_intent=False,
        confidence=0.94,
        trace_id="trace_hook_prep6",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep6",
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
    return prep1, prep2, prep3, prep4, prep5


def test_real_shared_slot_store_parity_harness_uses_isolated_store_only():
    *_rest, prep4, _prep5 = _prep_chain()
    parity = build_real_shared_slot_store_parity_harness(prep4, options=HGMQDTWritePrep6Options(max_tensor_dim=16))
    assert parity.parity_records
    assert parity.parity_ok is True
    assert parity.isolated_store_mutated is True
    assert parity.live_store_mutated is False
    record = parity.parity_records[0]
    assert record.observed_canonical_slot_id.startswith("css-")
    assert record.write_permission_granted is False
    assert record.parity_ok is True


def test_real_shared_slot_store_parity_invalid_input_fails_closed():
    parity = build_real_shared_slot_store_parity_harness(object())
    assert parity.validation.ok is False
    assert not parity.parity_records
    assert parity.parity_ok is False
    assert parity.live_store_mutated is False


def test_qh_storage_record_sandbox_constructs_validated_records():
    *_rest, prep4, _prep5 = _prep_chain()
    parity = build_real_shared_slot_store_parity_harness(prep4, options={"max_tensor_dim": 16})
    qh = build_qh_storage_record_sandbox(prep4, parity, options={"max_tensor_dim": 16})
    assert qh.records
    assert qh.constructed_count == len(qh.records)
    assert qh.validated_count == len(qh.records)
    assert qh.live_qh_storage_mutated is False
    rec = qh.records[0]
    assert rec.qh_record_id.startswith("qhrec-")
    assert rec.canonical_slot_id.startswith("css-")
    assert rec.write_permission_granted is False
    assert rec.validated is True


def test_qh_storage_record_sandbox_invalid_input_fails_closed():
    qh = build_qh_storage_record_sandbox(object())
    assert qh.validation.ok is False
    assert not qh.records
    assert qh.live_qh_storage_mutated is False


def test_rollback_snapshot_binding_dry_run_links_parity_and_qh_evidence():
    *_rest, prep4, _prep5 = _prep_chain()
    parity = build_real_shared_slot_store_parity_harness(prep4, options={"max_tensor_dim": 16})
    qh = build_qh_storage_record_sandbox(prep4, parity, options={"max_tensor_dim": 16})
    dry = build_rollback_snapshot_binding_dry_run(prep4, parity, qh)
    assert dry.bindings
    assert dry.live_rollback_stack_mutated is False
    assert all(not b.rollback_stack_mutated for b in dry.bindings)
    # Binding may remain not live-ready, but parity and qh evidence should be attached.
    assert dry.bindings[0].parity_evidence_id
    assert dry.bindings[0].qh_evidence_id


def test_rollback_snapshot_binding_invalid_input_fails_closed():
    dry = build_rollback_snapshot_binding_dry_run(object())
    assert dry.validation.ok is False
    assert not dry.bindings
    assert dry.binding_ready is False
    assert dry.live_rollback_stack_mutated is False


def test_high_level_write_prep_6_is_non_mutating():
    *_rest, prep4, prep5 = _prep_chain()
    result = build_hgm_qdt_write_prep_6(prep4, prep5, options=HGMQDTWritePrep6Options(max_tensor_dim=16))
    assert result.shared_slot_parity.parity_ok is True
    assert result.qh_sandbox.validated_count >= 1
    assert result.rollback_binding_dry_run.bindings
    assert result.metadata["live_write_executed"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["system_commitgate_commit_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False
    assert result.metadata["qh_storage_mutated"] is False
    assert result.metadata["rollback_stack_mutated"] is False
    assert result.metadata["production_write_ready"] is False


def test_high_level_write_prep_6_invalid_input_fails_closed():
    result = build_hgm_qdt_write_prep_6(object())
    assert result.validation.ok is False
    assert not result.shared_slot_parity.parity_records
    assert not result.qh_sandbox.records
    assert not result.rollback_binding_dry_run.bindings


def test_trace_redaction_does_not_leak_secret_metadata():
    *_rest, prep4, prep5 = _prep_chain()
    result = build_hgm_qdt_write_prep_6(prep4, prep5, options={"max_tensor_dim": 16})
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak-prep6" not in str(trace.redacted_payload())
