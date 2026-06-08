import torch

from mnemonic_cortex.ltm.hg_episodic_ltm import HGEpisodicLTM
from mnemonic_cortex.memory.memory_lifecycle_manager import MemoryLifecycleManager
from mnemonic_cortex.memory.memory_read_engine import MemoryReadEngine
from mnemonic_cortex.memory.memory_update_engine import MemoryUpdateEngine
from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine
from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_retention import SharedSlotRetention
from mnemonic_cortex.memory.shared_slot_schema import slot_state_to_code
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore


class _CaptureGeometryRuntime:
    def __init__(self):
        self.last_family = None
        self.last_depth = None

    def score(self, *, query, candidate_values, family=None, depth=None):
        self.last_family = family
        self.last_depth = depth
        # simple dot-product like score [B, C]
        if candidate_values.dim() == 3:
            return torch.einsum("bd,bcd->bc", query, candidate_values)
        return torch.einsum("bd,cd->bc", query, candidate_values)


class _GeometryPolicyRuntime:
    def resolve_family_depth(self, memory_type="episodic"):
        return "hyperbolic", 7


def _build_stack():
    store = SharedSlotStore(num_slots=96, slot_dim=8, num_systems=4, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    geometry_runtime = _CaptureGeometryRuntime()
    read_engine = MemoryReadEngine(store=store, geometry_runtime=geometry_runtime, reranker=None)
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    update_engine = MemoryUpdateEngine(store=store, write_engine=write_engine, arbitrator=arbitrator)
    retention = SharedSlotRetention(store=store)
    lifecycle = MemoryLifecycleManager(store=store, retention=retention)
    episodic = HGEpisodicLTM(
        slot_store=store,
        read_engine=read_engine,
        write_engine=write_engine,
        update_engine=update_engine,
        lifecycle=lifecycle,
        slot_dim=8,
        geometry_policy_runtime=_GeometryPolicyRuntime(),
        long_episode_threshold=6,
        summary_stride=3,
        promotion_retrieval_threshold=2,
        transformer_layers=0,
        fixed_transformer_layers=0,
        inherited_bank_layers=0,
        fusion_transformer_layers=0,
        decoder_transformer_layers=0,
        cross_model_attention_layers=0,
    )
    return episodic, store, geometry_runtime


def test_store_episode_uses_provisional_and_creates_summary_for_long_episode():
    episodic, store, _ = _build_stack()
    vectors = torch.randn(9, 8)
    rec = episodic.store_episode(
        episode_id="ep-001",
        episode_vectors=vectors,
        step_range=(10, 18),
        trace_ids=["trace-1"],
        tags=["event", "long"],
    )
    assert len(rec.slot_ids) == 9
    assert len(rec.summary_slot_ids) >= 2
    for sid in rec.slot_ids + rec.summary_slot_ids:
        assert int(store.slot_state_code[sid].item()) == slot_state_to_code("provisional")


def test_retrieve_uses_episodic_geometry_defaults_and_policy_family_depth():
    episodic, _, geometry_runtime = _build_stack()
    vectors = torch.randn(7, 8)
    episodic.store_episode(
        episode_id="ep-002",
        episode_vectors=vectors,
        step_range=(20, 26),
        trace_ids=["trace-2"],
        tags=["event"],
    )
    out = episodic.retrieve_episode_fragments(query=vectors[:1], top_k=4)
    assert out.diagnostics["used_geometry"] is True
    assert geometry_runtime.last_family == "hyperbolic"
    assert geometry_runtime.last_depth == 7
    assert out.values.shape[2] == 8


def test_repeated_retrieval_allows_lifecycle_promotion_to_durable():
    episodic, store, _ = _build_stack()
    vectors = torch.randn(6, 8)
    rec = episodic.store_episode(
        episode_id="ep-003",
        episode_vectors=vectors,
        step_range=(30, 35),
        trace_ids=["trace-3"],
        tags=["event"],
    )
    # First slots start provisional.
    sid = rec.slot_ids[0]
    assert int(store.slot_state_code[sid].item()) == slot_state_to_code("provisional")

    for _ in range(3):
        episodic.retrieve_episode_fragments(query=vectors[:1], top_k=3)

    assert int(store.slot_state_code[sid].item()) == slot_state_to_code("durable")
