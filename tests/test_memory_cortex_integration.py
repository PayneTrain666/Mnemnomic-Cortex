import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.memory import MemoryReadRequest, MemoryUpdateRequest, SlotWriteRequest


def test_cortex_shared_memory_subsystem_enable_and_io():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_shared_memory_subsystem(num_slots=64, num_systems=4, device=torch.device("cpu"), dtype=torch.float32)

    write_request = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[16],
        requested_state="volatile",
        requested_memory_type="episodic",
        confidence=0.8,
        provenance_trace_ids=["cortex:int:test"],
    )
    payload = torch.randn(2, 16)
    write_out = model.memory_write(write_request, payload)
    assert len(write_out.written_slot_ids) == 2

    read_request = MemoryReadRequest(
        requester_system="hg_mann",
        query=payload[:1],
        top_k=2,
        use_geometry=False,
    )
    read_out = model.memory_read(read_request)
    assert read_out.values.shape[0] == 1
    assert read_out.values.shape[2] == 16
    assert 1 <= read_out.values.shape[1] <= 2

    summary = model.memory_retention_summary(demotions=2, evictions=2)
    assert summary["enabled"] is True
    assert len(summary["demotion_candidates"]) <= 2
    assert len(summary["eviction_candidates"]) <= 2


def test_cortex_flush_memory_write_trace_pull_and_clear():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    assert model.flush_memory_write_trace() == []

    model.enable_shared_memory_subsystem(num_slots=32, num_systems=4, device=torch.device("cpu"), dtype=torch.float32)
    request = SlotWriteRequest(
        requester_system="hg_mann",
        candidate_value_shape=[16],
        requested_state="volatile",
        requested_memory_type="episodic",
        confidence=0.7,
        provenance_trace_ids=["trace:flush:test"],
    )
    model.memory_write(request, torch.randn(1, 16))
    events = model.flush_memory_write_trace()
    assert len(events) >= 1
    assert events[-1]["action"] in {"allocate_new", "overwrite", "merge"}
    assert model.flush_memory_write_trace() == []


def test_cortex_flush_memory_update_trace_pull_and_clear():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    assert model.flush_memory_update_trace() == []

    model.enable_shared_memory_subsystem(num_slots=32, num_systems=4, device=torch.device("cpu"), dtype=torch.float32)
    update_request = MemoryUpdateRequest(
        requester_system="hg_mann",
        slot_ids=[],
        new_values=torch.randn(1, 16),
        mode="merge",
        confidence=0.7,
        reason="trace flush integration",
    )
    model.memory_update(update_request)
    events = model.flush_memory_update_trace()
    assert len(events) >= 1
    assert events[-1]["action"] == "append_split"
    assert events[-1]["split_reason"] == "no_existing_targets"
    assert model.flush_memory_update_trace() == []
