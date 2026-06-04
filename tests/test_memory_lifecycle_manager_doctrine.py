import torch

from mnemonic_cortex.memory import (
    MemoryLifecycleManager,
    MemoryWriteEngine,
    SharedSlotAllocator,
    SharedSlotArbitrator,
    SharedSlotRetention,
    SlotProvenance,
    SlotWriteRequest,
    SharedSlotStore,
)


class _TruthRuntimeAllow:
    def allows_promotion(self, slot_id: int, diagnostics):
        return True

    def downstream_checks_pass(self, slot_id: int, diagnostics):
        return True


class _TruthRuntimeVetoPromotion:
    def allows_promotion(self, slot_id: int, diagnostics):
        return False

    def downstream_checks_pass(self, slot_id: int, diagnostics):
        return True


class _TruthRuntimeDownstreamFail:
    def allows_promotion(self, slot_id: int, diagnostics):
        return True

    def downstream_checks_pass(self, slot_id: int, diagnostics):
        return False


def _build_store():
    store = SharedSlotStore(num_slots=24, slot_dim=8, num_systems=4, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    writer = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    return store, writer


def _seed_slot(store: SharedSlotStore, writer: MemoryWriteEngine, *, state: str, usage: float, confidence: float, contradictions: int):
    request = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state=state,
        requested_memory_type="episodic",
        confidence=confidence,
        provenance_trace_ids=["trace:lifecycle:test"],
        extra={"explicit_requested_state": True} if state == "durable" else {},
    )
    out = writer.write(request=request, values=torch.randn(1, 8))
    slot_id = out.written_slot_ids[0]
    store.slot_usage[slot_id] = float(usage)
    meta = store.metadata[slot_id]
    if meta.provenance is None:
        meta.provenance = SlotProvenance(
            source_system="hg_mann",
            created_step=1,
            last_update_step=1,
            source_trace_ids=["trace:lifecycle:test"],
        )
    meta.provenance.contradiction_count = int(contradictions)
    store.metadata[slot_id] = meta
    return slot_id


def test_promote_to_durable_when_doctrine_conditions_hold():
    store, writer = _build_store()
    slot_id = _seed_slot(store, writer, state="volatile", usage=3.2, confidence=0.88, contradictions=0)
    retention = SharedSlotRetention(store=store)
    manager = MemoryLifecycleManager(store=store, retention=retention, truth_runtime=_TruthRuntimeAllow())

    decision = manager.evaluate_promotion(slot_id)
    assert decision.new_state == "durable"
    manager.apply_decision(decision)
    assert int(store.slot_state_code[slot_id].item()) == 3


def test_promotion_blocked_when_truth_runtime_objects():
    store, writer = _build_store()
    slot_id = _seed_slot(store, writer, state="provisional", usage=4.0, confidence=0.91, contradictions=0)
    retention = SharedSlotRetention(store=store)
    manager = MemoryLifecycleManager(store=store, retention=retention, truth_runtime=_TruthRuntimeVetoPromotion())

    decision = manager.evaluate_promotion(slot_id)
    assert decision.new_state == "provisional"
    assert decision.diagnostics["truth_maintenance_clear"] is False


def test_demote_durable_when_contradictions_and_usage_confidence_degrade():
    store, writer = _build_store()
    slot_id = _seed_slot(store, writer, state="durable", usage=0.3, confidence=0.2, contradictions=5)
    retention = SharedSlotRetention(store=store)
    manager = MemoryLifecycleManager(store=store, retention=retention, truth_runtime=_TruthRuntimeAllow())

    decision = manager.evaluate_demotion(slot_id)
    assert decision.new_state == "provisional"
    manager.apply_decision(decision)
    assert int(store.slot_state_code[slot_id].item()) == 2


def test_demote_provisional_when_downstream_checks_fail():
    store, writer = _build_store()
    slot_id = _seed_slot(store, writer, state="provisional", usage=2.0, confidence=0.7, contradictions=1)
    retention = SharedSlotRetention(store=store)
    manager = MemoryLifecycleManager(store=store, retention=retention, truth_runtime=_TruthRuntimeDownstreamFail())

    decision = manager.evaluate_demotion(slot_id)
    assert decision.new_state == "volatile"
    assert decision.diagnostics["downstream_checks_fail"] is True
