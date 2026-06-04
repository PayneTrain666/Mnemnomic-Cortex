from mnemonic_cortex.hypergraph_manifold import (
    BridgeAdapterStatus,
    DepthLayer,
    GeometryType,
    HGMBridgePayload,
    HGMQDTWritePrep2Options,
    HGMQDTWritePrep3Options,
    HGMQDTWritePrepOptions,
    SharedSlotLatticeHook,
    TraceSafeMemoryPlan,
    ValidationResult,
    build_hgm_qdt_write_prep_2,
    build_hgm_qdt_write_prep_3,
    build_hgm_qdt_write_prep_contracts,
    build_synthetic_shared_slot_store_sandbox,
    simulate_in_memory_commitgate,
    verify_rollback_replay,
)


def _sample_memory_plan():
    adapter = BridgeAdapterStatus(
        adapter_id="adapter_prep3",
        available=True,
        adapter_type="qdt_wm",
        reason="test adapter status",
        detected_paths=("mnemonic_cortex.working_memory",),
        trace_id="trace_adapter_prep3",
        metadata={},
    )
    payload = HGMBridgePayload(
        payload_id="payload_prep3",
        source_type="ProceduralActionSequence",
        source_id="seq_prep3",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        content_summary="prep3 procedural sequence summary",
        qspin_signature_id="qspin_placeholder_prep3",
        trace_id="trace_payload_prep3",
        metadata={"secret_token": "do-not-leak"},
    )
    hook = SharedSlotLatticeHook(
        hook_id="hook_prep3",
        source_record_id="payload_prep3",
        target_slot_id="hgm_slot_prep3",
        depth_layer=DepthLayer.D5_PROCEDURAL,
        geometry_type=GeometryType.SPCP,
        qspin_signature_id="qspin_placeholder_prep3",
        dry_run=True,
        write_intent=False,
        confidence=0.91,
        trace_id="trace_hook_prep3",
        metadata={},
    )
    return TraceSafeMemoryPlan(
        plan_id="plan_prep3",
        bridge_payloads=(payload,),
        slot_hooks=(hook,),
        adapter_status=adapter,
        dry_run=True,
        write_intent=False,
        validation=ValidationResult(),
        trace_records=tuple(),
        metadata={},
    )


def _prep2_result():
    prep1 = build_hgm_qdt_write_prep_contracts(_sample_memory_plan(), options=HGMQDTWritePrepOptions(proposal_dim=8))
    return build_hgm_qdt_write_prep_2(prep1, options=HGMQDTWritePrep2Options(max_tensor_dim=16))


def test_synthetic_shared_slot_store_sandbox_is_created_without_live_mutation():
    prep2 = _prep2_result()
    sandbox = build_synthetic_shared_slot_store_sandbox(prep2.proposal_builder_result.proposals)
    assert len(sandbox.slots) == 1
    assert sandbox.metadata["synthetic_only"] is True
    assert sandbox.metadata["live_store_mutated"] is False
    assert sandbox.slots[0].current_vector == tuple(0.0 for _ in range(8))


def test_in_memory_commitgate_simulation_uses_synthetic_store_only():
    prep2 = _prep2_result()
    sim = simulate_in_memory_commitgate(prep2)
    assert sim.operations
    assert sim.synthetic_store_mutated is True
    assert sim.live_store_mutated is False
    assert sim.stage_called is False
    assert sim.commit_called is False
    assert sim.metadata["system_commitgate_stage_called"] is False
    assert sim.metadata["system_commitgate_commit_called"] is False


def test_invalid_input_fails_closed():
    sim = simulate_in_memory_commitgate(object())
    assert sim.validation.ok is False
    assert not sim.operations
    assert sim.live_store_mutated is False


def test_bounded_operation_count_is_enforced():
    prep2 = _prep2_result()
    proposals = prep2.proposal_builder_result.proposals * 4
    sim = simulate_in_memory_commitgate(proposals, options={"max_operations": 2})
    assert len(sim.operations) == 2
    assert sim.validation.warnings


def test_rollback_replay_restores_synthetic_previous_state():
    prep2 = _prep2_result()
    sim = simulate_in_memory_commitgate(prep2)
    rollback = verify_rollback_replay(sim)
    assert rollback.replay_records
    assert rollback.verified is True
    assert rollback.synthetic_store_restored is True
    assert rollback.live_rollback_stack_mutated is False
    for record in rollback.replay_records:
        assert record.restored is True


def test_synthetic_commit_can_be_disabled_without_live_write():
    prep2 = _prep2_result()
    sim = simulate_in_memory_commitgate(prep2, options={"allow_synthetic_commit": False})
    assert sim.operations
    assert all(not op.committed for op in sim.operations)
    assert sim.synthetic_store_mutated is False
    assert sim.live_store_mutated is False


def test_high_level_write_prep_3_returns_isolated_result():
    prep2 = _prep2_result()
    result = build_hgm_qdt_write_prep_3(prep2)
    assert result.commitgate_simulation.operations
    assert result.rollback_replay.verified is True
    assert result.metadata["live_write_executed"] is False
    assert result.metadata["system_commitgate_stage_called"] is False
    assert result.metadata["system_commitgate_commit_called"] is False
    assert result.metadata["shared_slot_store_mutated"] is False
    assert result.metadata["rollback_stack_mutated"] is False


def test_trace_redaction_does_not_leak_secret_metadata():
    prep2 = _prep2_result()
    result = build_hgm_qdt_write_prep_3(prep2)
    assert result.trace_records
    for trace in result.trace_records:
        assert "do-not-leak" not in str(trace.redacted_payload())


def test_rollback_invalid_simulation_fails_closed():
    rollback = verify_rollback_replay(object())
    assert rollback.validation.ok is False
    assert rollback.verified is False
    assert rollback.live_rollback_stack_mutated is False
