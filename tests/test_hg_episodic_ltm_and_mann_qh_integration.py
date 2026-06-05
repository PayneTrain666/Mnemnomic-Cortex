import torch

from mnemonic_cortex.ltm.hg_episodic_ltm import HGEpisodicLTM
from mnemonic_cortex.memory.memory_lifecycle_manager import MemoryLifecycleManager
from mnemonic_cortex.memory.memory_read_engine import MemoryReadEngine
from mnemonic_cortex.memory.memory_update_engine import MemoryUpdateEngine
from mnemonic_cortex.memory.memory_write_engine import MemoryWriteEngine
from mnemonic_cortex.memory.shared_slot_allocator import SharedSlotAllocator
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_retention import SharedSlotRetention
from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore
from mnemonic_cortex.reasoning_depth.mann_slotkv_depth_bank import (
    MANNSlotKVDepthBank,
    MANNSlotKVDepthBankConfig,
)


class _GeometryRuntime:
    def score(self, *, query, candidate_values, family=None, depth=None):
        if candidate_values.dim() == 3:
            return torch.einsum("bd,bcd->bc", query, candidate_values)
        return torch.einsum("bd,cd->bc", query, candidate_values)


def _build_episodic():
    store = SharedSlotStore(num_slots=64, slot_dim=8, num_systems=4, device="cpu", dtype=torch.float32)
    allocator = SharedSlotAllocator(store=store)
    arbitrator = SharedSlotArbitrator(store=store)
    read_engine = MemoryReadEngine(store=store, geometry_runtime=_GeometryRuntime(), reranker=None)
    write_engine = MemoryWriteEngine(store=store, allocator=allocator, arbitrator=arbitrator)
    update_engine = MemoryUpdateEngine(store=store, write_engine=write_engine, arbitrator=arbitrator)
    lifecycle = MemoryLifecycleManager(store=store, retention=SharedSlotRetention(store=store))
    return HGEpisodicLTM(
        slot_store=store,
        read_engine=read_engine,
        write_engine=write_engine,
        update_engine=update_engine,
        lifecycle=lifecycle,
        slot_dim=8,
        long_episode_threshold=6,
        summary_stride=3,
    )


def test_hg_episodic_ltm_writes_qh_holograms_with_episode_storage():
    episodic = _build_episodic()
    rec = episodic.store_episode(
        episode_id="qh-ep-1",
        episode_vectors=torch.randn(9, 8),
        step_range=(4, 12),
        trace_ids=["tr-a", "tr-b"],
        tags=["episodic", "qh"],
    )
    qh = episodic.qh_trace_summary()
    assert qh["triplets_stored"] > 0
    assert qh["active_slots"] > 0
    first_meta = episodic.slot_store.get_slot_metadata([rec.slot_ids[0]])[0]
    assert bool(first_meta.extra.get("qh_hologram")) is True


def test_mann_slotkv_depth_bank_records_qh_shadow_holograms():
    bank = MANNSlotKVDepthBank(
        MANNSlotKVDepthBankConfig.enabled_default(key_dim=16, value_dim=16, slot_count=16)
    )
    out = bank.propose_hop_writes(
        slot_index=3,
        value=torch.randn(16),
        key=torch.randn(16),
        canonical_slot_id="mann-qh-shadow",
        hop_id=1,
    )
    assert out["metadata"]["qh_shadow_holograms_recorded"] is True
    assert out["metadata"]["qh_shadow_stats"]["stored"] > 0
    caps = bank.capacity_metrics()
    assert "qh_shadow" in caps
    assert caps["qh_shadow"]["triplets_stored"] > 0
