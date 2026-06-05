import torch

from mnemonic_cortex.memory.memory_update_engine import MemoryUpdateEngine, MemoryUpdateRequest
from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine
from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_schema import SlotMetadata, SlotProvenance, SlotWriteRequest
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore


def _build_stack():
    store = SharedSlotStore(num_slots=12, slot_dim=8, num_systems=3, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    update_engine = MemoryUpdateEngine(store=store, write_engine=write_engine, arbitrator=arbitrator)
    return store, write_engine, update_engine


def _seed_slot(store: SharedSlotStore, state: str = "volatile", semantic_tags=None, contradiction_count: int = 0) -> int:
    req = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state="volatile",
        requested_memory_type="episodic",
        confidence=0.6,
    )
    from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
    from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
    from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine

    writer = MemoryWriteEngine(
        store=store,
        allocator=SharedSlotAllocator(store=store),
        arbitrator=SharedSlotArbitrator(store=store),
    )
    out = writer.write(request=req, values=torch.randn(1, 8))
    slot_id = out.written_slot_ids[0]
    meta = store.metadata[slot_id]
    meta.state = state
    meta.semantic_tags = list(semantic_tags or [])
    if meta.provenance is None:
        meta.provenance = SlotProvenance(
            source_system="hg_mann",
            created_step=1,
            last_update_step=1,
        )
    meta.provenance.contradiction_count = int(contradiction_count)
    store.metadata[slot_id] = meta
    return slot_id


def test_split_doctrine_high_contradiction_uses_append_split():
    store, _, updater = _build_stack()
    slot_id = _seed_slot(store, contradiction_count=5, semantic_tags=["facts"])

    request = MemoryUpdateRequest(
        requester_system="hg_mann",
        slot_ids=[slot_id],
        new_values=torch.randn(1, 8),
        mode="merge",
        semantic_tags=["facts"],
        reason="conflict observed",
    )
    out = updater.update(request)
    written_meta = store.metadata[out.written_slot_ids[0]]
    assert written_meta.extra["split_doctrine_reason"] == "high_contradiction_burden"
    assert out.written_slot_ids[0] != slot_id


def test_split_doctrine_semantic_incompatibility_uses_append_split():
    store, _, updater = _build_stack()
    slot_id = _seed_slot(store, contradiction_count=0, semantic_tags=["physics"])

    request = MemoryUpdateRequest(
        requester_system="hg_mann",
        slot_ids=[slot_id],
        new_values=torch.randn(1, 8),
        mode="overwrite",
        semantic_tags=["biology"],
        reason="new domain material",
    )
    out = updater.update(request)
    written_meta = store.metadata[out.written_slot_ids[0]]
    assert written_meta.extra["split_doctrine_reason"] == "semantic_incompatible"
    assert out.written_slot_ids[0] != slot_id


def test_split_doctrine_durable_slot_protects_from_overwrite():
    store, _, updater = _build_stack()
    slot_id = _seed_slot(store, state="durable", contradiction_count=0, semantic_tags=["episodic"])

    request = MemoryUpdateRequest(
        requester_system="hg_mann",
        slot_ids=[slot_id],
        new_values=torch.randn(1, 8),
        mode="merge",
        semantic_tags=["episodic"],
        reason="routine refresh",
    )
    out = updater.update(request)
    written_meta = store.metadata[out.written_slot_ids[0]]
    assert written_meta.extra["split_doctrine_reason"] == "durable_slot_protected"
    assert out.written_slot_ids[0] != slot_id
