from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrep2Options,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_dry_run_system_write_proposal_previews,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_contracts,
    run_commitgate_preflight,
    simulate_end_to_end_hgm_qdt_write,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep2",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep2",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep2",
        source_type="ProceduralActionSequence",
        source_id="seq_prep2",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep2 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep2",
        trace_id="trace_payload_prep2",
        metadata={"secret_token": "do-not-leak"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep2",
        source_record_id="payload_prep2",
        target_slot_id="hgm_slot_prep2",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep2",
        dry_run=True,
        write_intent=False,
        confidence=0.91,
        trace_id="trace_hook_prep2",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep2",
        bridge_payloads=(payload,),
        slot_hooks=(hook,),
        adapter_status=adapter,
        dry_run=True,
        write_intent=False,
        validation=ValidationResult(),
        trace_records=tuple(),
        metadata={},
    )


def _write_prep_1_result():
    return build_hgm_qdt_write_prep_contracts(_sample_memory_plan(), options=HGMQDTWritePrepOptions(proposal_dim=8))


def test_dry_run_system_write_proposal_preview_builder_creates_non_mutating_preview():
    prep1 = _write_prep_1_result()
    result = build_dry_run_system_write_proposal_previews(prep1, options=HGMQDTWritePrep2Options(max_tensor_dim=16))
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.proposal_id.startswith("drysysprop-")
    assert proposal.content_shape == (8,)
    assert proposal.write_permission is False
    assert proposal.metadata["stage_called"] is False
    assert proposal.metadata["commit_called"] is False
    assert result.metadata["live_writes"] is False


def test_missing_slot_mapping_blocks_proposal_preview_readiness():
    prep1 = _write_prep_1_result()
    bad = type("BadPrep", (), {
        "proposal_contract": prep1.proposal_contract,
        "slot_mapping_plan": None,
        "qspin_qh_contract": prep1.qspin_qh_contract,
        "rollback_handshake": prep1.rollback_handshake,
    })()
    result = build_dry_run_system_write_proposal_previews(bad)
    assert result.validation.ok is False
    assert any("slot" in m.code for m in result.validation.errors)


def test_missing_qh_conversion_blocks_qh_ready_status():
    prep1 = _write_prep_1_result()
    bad = type("BadPrep", (), {
        "proposal_contract": prep1.proposal_contract,
        "slot_mapping_plan": prep1.slot_mapping_plan,
        "qspin_qh_contract": None,
        "rollback_handshake": prep1.rollback_handshake,
    })()
    result = build_dry_run_system_write_proposal_previews(bad)
    assert result.validation.ok is False
    assert any("qh" in m.code.lower() for m in result.validation.errors)


def test_commitgate_preflight_does_not_stage_or_commit():
    prep1 = _write_prep_1_result()
    builder = build_dry_run_system_write_proposal_previews(prep1)
    preflight = run_commitgate_preflight(builder, write_prep_result=prep1)
    assert preflight.checks
    assert preflight.metadata["stage_called"] is False
    assert preflight.metadata["commit_called"] is False
    assert any(c.check_name == "no_stage_or_commit" and c.passed for c in preflight.checks)


def test_missing_rollback_handshake_blocks_commit_simulation_readiness():
    prep1 = _write_prep_1_result()
    report = simulate_end_to_end_hgm_qdt_write(prep1)
    assert report.rollback_ready is False
    assert report.readiness_score <= HGMQDTWritePrep2Options().conservative_missing_score
    assert report.live_write_executed is False
    assert report.stage_called is False
    assert report.commit_called is False


def test_simulated_write_permission_is_dry_run_only():
    prep1 = _write_prep_1_result()
    result = build_dry_run_system_write_proposal_previews(prep1, options={"simulated_write_permission": True})
    proposal = result.proposals[0]
    assert proposal.simulated_write_permission is True
    assert proposal.write_permission is False
    assert result.metadata["live_writes"] is False


def test_invalid_contract_fails_closed():
    result = build_dry_run_system_write_proposal_previews(object())
    assert result.validation.ok is False
    assert not result.proposals


def test_high_level_write_prep_2_returns_trace_safe_report():
    prep1 = _write_prep_1_result()
    result = build_hgm_qdt_write_prep_2(prep1)
    assert result.proposal_builder_result.proposals
    assert result.preflight_result.checks
    assert result.simulation_report.metadata["no_system_commit_gate_stage"] is True
    assert result.metadata["live_write_executed"] is False
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak" not in str(trace.redacted_payload())


def test_bounded_proposal_count_is_enforced():
    plan = _sample_memory_plan()
    # Duplicate payload/hook records to create multiple tensor previews through prep1.
    payloads = tuple(plan.bridge_payloads * 4)
    hooks = tuple(plan.slot_hooks * 4)
    plan2 = TraceSafeMemoryPlan(
        plan_id="plan_many_prep2",
        bridge_payloads=payloads,
        slot_hooks=hooks,
        adapter_status=plan.adapter_status,
        dry_run=True,
        write_intent=False,
        validation=ValidationResult(),
    )
    prep1 = build_hgm_qdt_write_prep_contracts(plan2, options=HGMQDTWritePrepOptions(proposal_dim=4))
    result = build_dry_run_system_write_proposal_previews(prep1, options={"max_proposals": 2})
    assert len(result.proposals) == 2
    assert result.validation.warnings
