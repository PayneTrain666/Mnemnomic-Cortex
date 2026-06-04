from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_hgm_qdt_write_prep_contracts,
    build_proposal_materialization_contract,
    build_qspin_qh_conversion_contract,
    build_rollback_snapshot_handshake,
    build_slot_id_mapping_plan,
    canonical_css_preview,
    depth_to_qdt_index,
    geometry_to_qdt_map,
    probe_qdt_wm_write_contracts,
    sanitize_hgm_slot_id_for_wm,
)


def _sample_memory_plan():
    validation = ValidationResult()
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_test",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_alpha",
        source_type="ProceduralActionSequence",
        source_id="seq_alpha",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="test procedural sequence summary",
        qspin_signature_id="qspin_placeholder_alpha",
        trace_id="trace_payload",
        metadata={"secret_token": "do-not-leak"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_alpha",
        source_record_id="payload_alpha",
        target_slot_id="hgm_slot_alpha",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_alpha",
        dry_run=True,
        write_intent=False,
        confidence=0.9,
        trace_id="trace_hook",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_alpha",
        bridge_payloads=(payload,),
        slot_hooks=(hook,),
        adapter_status=adapter,
        dry_run=True,
        write_intent=False,
        validation=validation,
        trace_records=tuple(),
        metadata={},
    )


def test_signature_level_probe_returns_contract_records():
    result = probe_qdt_wm_write_contracts()
    assert result.probes
    names = {p.symbol_name for p in result.probes}
    assert "SystemWriteProposal" in names
    assert "canonical_slot_id" in names
    assert result.validation.ok or result.validation.warnings


def test_proposal_materialization_contract_creates_bounded_tensor_preview():
    plan = _sample_memory_plan()
    result = build_proposal_materialization_contract(plan, options=HGMQDTWritePrepOptions(proposal_dim=8))
    assert len(result.tensor_previews) == 1
    preview = result.tensor_previews[0]
    assert preview.content_shape == (8,)
    assert preview.finite is True
    assert preview.bounded is True
    assert preview.write_permission is False


def test_slot_id_mapping_plan_maps_hgm_to_wm_and_css():
    plan = _sample_memory_plan()
    result = build_slot_id_mapping_plan(plan, namespace="test_hgm")
    assert len(result.mappings) == 1
    mapping = result.mappings[0]
    assert mapping.wm_local_slot_id.startswith("wm_")
    assert mapping.wm_canonical_slot_id.startswith("css-")
    assert sanitize_hgm_slot_id_for_wm("hgm_slot_x").startswith("wm_")
    assert canonical_css_preview("test", "wm_x").startswith("css-")


def test_qspin_qh_conversion_contract_maps_depth_geometry_and_placeholder():
    plan = _sample_memory_plan()
    result = build_qspin_qh_conversion_contract(plan)
    assert result.conversions
    first = result.conversions[0]
    assert first.depth_index == 5
    assert first.geometry_map == "spcp"
    assert first.qh_record_id_preview.startswith("qhrec-")
    assert depth_to_qdt_index(DepthLayer.D4_CAUSAL) == 4
    assert geometry_to_qdt_map(GeometryType.COMPLEX_PROJECTIVE) == "complex_projective"
    assert any("placeholder" in msg.code or "placeholder" in msg.message for msg in result.validation.warnings)


def test_rollback_snapshot_handshake_is_not_complete_without_real_snapshots():
    plan = _sample_memory_plan()
    result = build_rollback_snapshot_handshake(plan)
    assert len(result.requirements) == 1
    assert result.complete is False
    assert result.requirements[0].bound_to_actual_snapshot is False
    assert "rollback_stack" in result.requirements[0].snapshot_source


def test_high_level_write_prep_pipeline_is_read_only_and_traceable():
    plan = _sample_memory_plan()
    result = build_hgm_qdt_write_prep_contracts(plan, options=HGMQDTWritePrepOptions(proposal_dim=8))
    assert result.proposal_contract.tensor_previews
    assert result.slot_mapping_plan.mappings
    assert result.qspin_qh_contract.conversions
    assert result.rollback_handshake.requirements
    assert result.metadata["read_only"] is True
    assert result.metadata["live_writes"] is False
    assert result.trace_records


def test_invalid_memory_plan_fails_closed():
    result = build_hgm_qdt_write_prep_contracts(object())
    assert result.validation.ok is False
    assert any(m.code == "write_prep.invalid_memory_plan" for m in result.validation.errors)
