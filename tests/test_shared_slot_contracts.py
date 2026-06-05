import torch

from mnemonic_cortex.memory import (
    MemoryReadEngine,
    MemoryReadRequest,
    MemoryWriteEngine,
    SharedSlotAllocator,
    SharedSlotArbitrator,
    SharedSlotStore,
    SlotWriteRequest,
)


def _build_stack(num_slots: int = 32):
    store = SharedSlotStore(num_slots=num_slots, slot_dim=16, num_systems=8, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    writer = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    reader = MemoryReadEngine(store=store, arbitrator=arbitrator)
    return store, allocator, arbitrator, writer, reader


def test_write_request_validation_rejects_bad_system():
    req = SlotWriteRequest(
        requester_system="unknown_system",
        candidate_value_shape=[1, 16],
    )
    try:
        req.validate()
        assert False, "expected ValueError for unknown requester_system"
    except ValueError:
        pass


def test_write_updates_primary_system_code_and_acl_masks():
    store, _, _, writer, _ = _build_stack()
    req = SlotWriteRequest(
        requester_system="hg_ep_ltm",
        candidate_value_shape=[1, 16],
        requested_state="provisional",
        confidence=0.7,
    )
    out = writer.write(request=req, values=torch.randn(1, 16))
    sid = out.written_slot_ids[0]
    assert int(store.primary_system_code[sid].item()) == 0
    assert bool(store.allowed_read_mask[sid, 0].item()) is True
    assert bool(store.allowed_write_mask[sid, 0].item()) is True


def test_read_engine_reports_read_arbitrator_usage():
    _, _, _, writer, reader = _build_stack()
    req = SlotWriteRequest(
        requester_system="hg_ep_ltm",
        candidate_value_shape=[1, 16],
        requested_state="provisional",
    )
    writer.write(request=req, values=torch.randn(1, 16))
    out = reader.retrieve(
        MemoryReadRequest(
            requester_system="hg_ep_ltm",
            query=torch.randn(1, 16),
            top_k=4,
        )
    )
    assert out.diagnostics.get("used_read_arbitrator") is True
