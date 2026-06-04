import torch

from mnemonic_cortex.memory.memory_read_engine import MemoryReadEngine, MemoryReadRequest
from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine
from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_schema import SlotWriteRequest
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore


class DotGeometryRuntime:
    def score(self, *, query, candidate_values, family=None, depth=None):
        if candidate_values.dim() == 2:
            candidate_values = candidate_values.unsqueeze(0).expand(query.size(0), -1, -1)
        return torch.einsum("bd,bcd->bc", query, candidate_values)


def test_end_to_end_write_then_read_pipeline():
    torch.manual_seed(0)
    store = SharedSlotStore(num_slots=24, slot_dim=8, num_systems=3, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    read_engine = MemoryReadEngine(store=store, geometry_runtime=DotGeometryRuntime(), reranker=None)

    payload = torch.randn(3, 8)
    write_request = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[8],
        requested_state="volatile",
        requested_memory_type="episodic",
        confidence=0.8,
        provenance_trace_ids=["pipe:1"],
    )
    write_out = write_engine.write(request=write_request, values=payload)
    assert len(write_out.written_slot_ids) == 3

    query = payload[0].unsqueeze(0)  # [B=1, D]
    read_request = MemoryReadRequest(
        requester_system="hg_mann",
        query=query,
        top_k=2,
        use_geometry=True,
    )
    read_out = read_engine.retrieve(read_request)

    assert read_out.slot_ids.shape == (1, 2)
    assert read_out.scores.shape == (1, 2)
    assert read_out.values.shape == (1, 2, 8)
    # At least one retrieved slot should be from those we wrote.
    assert any(int(slot_id) in set(write_out.written_slot_ids) for slot_id in read_out.slot_ids[0].tolist())
