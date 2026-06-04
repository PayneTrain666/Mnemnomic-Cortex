import pytest
import torch

from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine
from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_schema import SlotMetadata, SlotWriteRequest
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore


def _build_stack():
    store = SharedSlotStore(num_slots=12, slot_dim=8, num_systems=3, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    return store, allocator, arbitrator, engine


def test_single_vector_write_is_normalized_and_written():
    store, _, _, engine = _build_stack()
    request = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state="volatile",
        confidence=0.7,
    )
    vec = torch.randn(8)
    out = engine.write(request=request, values=vec)

    assert len(out.written_slot_ids) == 1
    slot_id = out.written_slot_ids[0]
    assert store.get_slot_value([slot_id]).shape == (1, 8)
    assert out.diagnostics["input_shape"] == [1, 8]


def test_write_path_always_logs_provenance():
    store, _, _, engine = _build_stack()
    request = SlotWriteRequest(
        requester_system="semantic_mann",
        candidate_value_shape=[8],
        requested_state="provisional",
        confidence=0.55,
        provenance_trace_ids=["trace-a", "trace-b"],
    )
    values = torch.randn(1, 8)
    out = engine.write(request=request, values=values)

    meta = store.metadata[out.written_slot_ids[0]]
    assert isinstance(meta, SlotMetadata)
    assert meta.provenance is not None
    assert meta.provenance.source_system == "semantic_mann"
    assert meta.provenance.source_trace_ids == ["trace-a", "trace-b"]
    assert out.diagnostics["provenance_logged"] is True
    trace = engine.get_write_trace()
    assert trace, "write trace should record events"
    assert trace[-1]["action"] in {"allocate_new", "overwrite", "merge"}
    assert trace[-1]["requester_system"] == "semantic_mann"


def test_durable_write_requires_explicit_state_or_lifecycle_approval():
    _, _, _, engine = _build_stack()
    values = torch.randn(1, 8)

    denied = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state="durable",
        confidence=0.9,
        extra={},  # no explicit/lifecycle flags
    )
    with pytest.raises(PermissionError):
        engine.write(request=denied, values=values)

    allowed_explicit = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state="durable",
        confidence=0.9,
        extra={"explicit_requested_state": True},
    )
    out = engine.write(request=allowed_explicit, values=values)
    assert len(out.written_slot_ids) == 1
