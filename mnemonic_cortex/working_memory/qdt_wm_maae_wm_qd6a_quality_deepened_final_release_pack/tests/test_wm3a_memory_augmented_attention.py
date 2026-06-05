import torch

from mnemonic_cortex.working_memory import (
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    WMMemoryAugmentedAttentionConfig,
    WMMemoryAugmentedAttention,
    QDTWorkingMemoryConfig,
    QDTWorkingMemory,
)


def test_memory_augmented_attention_shape_trace_and_policy_lane():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=8, dim=32))
    maae = WMMemoryAugmentedAttention(WMMemoryAugmentedAttentionConfig(dim=32, top_k=3), slot_bank=bank)
    tokens = torch.randn(2, 5, 32)

    out, trace = maae(tokens, require_write_permission=True, return_trace=True)

    assert out.shape == tokens.shape
    assert torch.isfinite(out).all()
    assert trace["memory_context_shape"] == [2, 32]
    assert trace["paamax_metadata"]["policy_lane_present"] is True
    assert trace["paamax_metadata"]["write_permission_required"] is True
    assert trace["paamax_metadata"]["write_permission_granted"] is True


def test_memory_augmented_attention_stability_report():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=8, dim=32))
    maae = WMMemoryAugmentedAttention(WMMemoryAugmentedAttentionConfig(dim=32, top_k=3), slot_bank=bank)
    tokens = torch.randn(2, 5, 32)

    report = maae.stability_report(tokens)
    assert report["ok"] is True
    assert report["finite"] is True
    assert report["shape_ok"] is True
    assert report["policy_lane_present"] is True


def test_qdt_working_memory_read_includes_memory_augmented_attention_trace():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = wm(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    stages = [item["stage"] for item in trace["items"]]
    assert "memory_augmented_attention" in stages
    maae_items = [item for item in trace["items"] if item["stage"] == "memory_augmented_attention"]
    assert maae_items
    payload = maae_items[-1]["metadata"]["payload"]
    assert payload["paamax_metadata"]["policy_lane_present"] is True


def test_memory_augmented_attention_rejects_bad_shape():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=8, dim=32))
    maae = WMMemoryAugmentedAttention(WMMemoryAugmentedAttentionConfig(dim=32, top_k=3), slot_bank=bank)
    bad = torch.randn(2, 32)
    try:
        maae(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
