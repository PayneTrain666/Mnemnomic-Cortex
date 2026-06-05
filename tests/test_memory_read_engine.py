import torch
import torch.nn as nn

from mnemonic_cortex.memory.memory_read_engine import MemoryReadEngine, MemoryReadRequest
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore


class DummyGeometryRuntime:
    def score(self, *, query, candidate_values, family=None, depth=None):
        # query: [B, D], candidate_values: [B, C, D]
        if candidate_values.dim() == 2:
            candidate_values = candidate_values.unsqueeze(0).expand(query.size(0), -1, -1)
        # simple dot-product geometry surrogate
        return torch.einsum("bd,bcd->bc", query, candidate_values)


class DummyReranker(nn.Module):
    def forward(self, query, candidate_values, candidate_scores):
        # Mildly prefer lower index by subtracting tiny rank offset.
        c = candidate_scores.size(1)
        offset = torch.linspace(0.0, 1e-3, c, device=candidate_scores.device, dtype=candidate_scores.dtype)
        return candidate_scores - offset.unsqueeze(0)


def _seeded_store() -> SharedSlotStore:
    torch.manual_seed(0)
    store = SharedSlotStore(num_slots=16, slot_dim=8, num_systems=3, device="cpu", dtype=torch.float32)
    slot_ids = torch.arange(0, 10, dtype=torch.long)
    state_code = torch.tensor([1, 1, 2, 3, 1, 2, 3, 1, 4, 5], dtype=torch.long)
    store.set_slot_state_code(slot_ids=slot_ids, state_code=state_code)
    values = torch.randn(10, 8)
    store.set_slot_value(slot_ids=slot_ids, values=values, confidence=torch.linspace(0.2, 0.95, 10))
    return store


def test_memory_read_engine_shapes_and_dependencies():
    store = _seeded_store()
    engine = MemoryReadEngine(
        store=store,
        geometry_runtime=DummyGeometryRuntime(),
        reranker=DummyReranker(),
    )

    query = torch.randn(2, 8)
    request = MemoryReadRequest(requester_system="hg_mann", query=query, top_k=4, use_geometry=True)
    out = engine.retrieve(request)

    assert out.slot_ids.shape == (2, 4)
    assert out.scores.shape == (2, 4)
    assert out.values.shape == (2, 4, 8)
    assert len(out.metadata) == 2
    assert all(len(row) == 4 for row in out.metadata)

    # Confirm shape diagnostics align with expected substrate/retrieval flow.
    assert out.diagnostics["global_shape"] == [16, 8]  # [N, D]
    assert out.diagnostics["candidate_shape"][1] == 8  # [C, D]
    assert out.diagnostics["batched_candidate_shape"][0] == 2  # [B, C, D]
    assert out.diagnostics["output_shape"] == [2, 4, 8]  # [B, K, D]
    assert out.diagnostics["used_geometry"] is True
    assert out.diagnostics["used_reranker"] is True
