import torch

from mnemonic_cortex.working_memory import (
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    WMMemoryAugmentedAttentionConfig,
    WMMemoryAugmentedAttention,
    QDTWorkingMemoryConfig,
    QDTWorkingMemory,
)


def test_memory_augmented_attention_contains_advanced_attention_traces():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=8, dim=32))
    module = WMMemoryAugmentedAttention(WMMemoryAugmentedAttentionConfig(dim=32, top_k=3), slot_bank=bank)
    tokens = torch.randn(2, 5, 32)
    prior_trace = {"items": [{"stage": "curved_core"}], "confidence": 0.8, "disagreement": 0.1, "paamax_metadata": {"policy_lane_present": True}}

    y, trace = module(tokens, prior_trace=prior_trace, return_trace=True)

    assert y.shape == tokens.shape
    adv = trace["advanced_attention_traces"]
    expected = {
        "evidence_attention",
        "trace_attention",
        "counterfactual_attention",
        "conflict_attention",
        "novelty_attention",
        "stability_attention",
    }
    assert expected.issubset(set(adv))
    assert trace["paamax_metadata"]["advanced_attention_present"] is True
    assert trace["paamax_metadata"]["stability_guard"] is True


def test_qdt_working_memory_trace_contains_advanced_attention_metadata():
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(2, 5, 32)
    y, trace = wm(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    maae_items = [item for item in trace["items"] if item["stage"] == "memory_augmented_attention"]
    assert maae_items
    payload = maae_items[-1]["metadata"]["payload"]
    assert payload["paamax_metadata"]["advanced_attention_present"] is True
    assert "evidence_attention" in payload["advanced_attention_traces"]
    assert "conflict_attention" in payload["advanced_attention_traces"]
    assert "stability_attention" in payload["advanced_attention_traces"]
