import torch

from mnemonic_cortex.memory import SharedSlotRetention, SharedSlotStore
from mnemonic_cortex.memory.shared_slot_schema import SlotMetadata, SlotProvenance


def _make_store() -> SharedSlotStore:
    store = SharedSlotStore(num_slots=8, slot_dim=4, num_systems=3, device="cpu", dtype=torch.float32)
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    store.set_slot_state_code(slot_ids=ids, state_code=torch.tensor([1, 1, 3, 5], dtype=torch.long))
    store.slot_usage[ids] = torch.tensor([2.0, 4.0, 6.0, 2.0])
    store.slot_confidence[ids] = torch.tensor([0.3, 0.7, 0.9, 0.4])
    store.slot_age[ids] = torch.tensor([10, 40, 120, 50], dtype=torch.long)

    store.metadata[0] = SlotMetadata(
        slot_id=0,
        state="volatile",
        primary_system_id="hg_mann",
        provenance=SlotProvenance(source_system="hg_mann", created_step=0, last_update_step=10),
    )
    store.metadata[1] = SlotMetadata(
        slot_id=1,
        state="volatile",
        primary_system_id="hg_mann",
        provenance=SlotProvenance(
            source_system="hg_mann",
            created_step=0,
            last_update_step=10,
            promotion_count=2,
        ),
    )
    store.metadata[2] = SlotMetadata(
        slot_id=2,
        state="durable",
        primary_system_id="semantic_mann",
        provenance=SlotProvenance(
            source_system="semantic_mann",
            created_step=0,
            last_update_step=10,
            promotion_count=6,
            contradiction_count=0,
        ),
    )
    store.metadata[3] = SlotMetadata(
        slot_id=3,
        state="quarantined",
        primary_system_id="semantic_mann",
        provenance=SlotProvenance(
            source_system="semantic_mann",
            created_step=0,
            last_update_step=10,
            promotion_count=0,
            contradiction_count=6,
        ),
    )
    return store


def test_retention_scores_are_deterministic():
    store = _make_store()
    retention = SharedSlotRetention(store=store)

    a = retention.score_all_active_slots()
    b = retention.score_all_active_slots()

    assert [x.slot_id for x in a] == [x.slot_id for x in b]
    assert [round(x.keep_score, 8) for x in a] == [round(x.keep_score, 8) for x in b]
    assert [round(x.evict_score, 8) for x in a] == [round(x.evict_score, 8) for x in b]


def test_retention_state_bias_prefers_durable_over_quarantined():
    store = _make_store()
    retention = SharedSlotRetention(store=store)

    durable = retention.score_slot(2)
    quarantined = retention.score_slot(3)

    assert durable.keep_score > quarantined.keep_score
    assert quarantined.evict_score > durable.evict_score
