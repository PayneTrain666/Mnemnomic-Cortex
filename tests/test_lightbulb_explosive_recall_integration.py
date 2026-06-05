import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.memory_curved import EnhancedCurvedMemory


def test_curved_memory_read_accepts_explosive_recall_kwargs():
    torch.manual_seed(0)
    mem = EnhancedCurvedMemory(input_dim=32, mem_slots=64, topk=8)
    x = torch.randn(3, 5, 32)
    fire = torch.tensor([True, False, True])
    out = mem(x, operation="read", fire_mask=fire, recall_boost=0.8)
    assert out.shape == x.shape


def test_associative_retrieve_changes_under_explosive_recall():
    torch.manual_seed(7)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    cue = torch.randn(2, 4, 16)
    context = torch.randn(2, 16)
    base = model.retrieve_memory(
        cue,
        context,
        strategy="associative",
        fire_mask=torch.zeros(2, dtype=torch.bool),
        recall_boost=0.0,
    )
    boosted = model.retrieve_memory(
        cue,
        context,
        strategy="associative",
        fire_mask=torch.ones(2, dtype=torch.bool),
        recall_boost=0.9,
    )
    assert base.shape == boosted.shape == (2, 16)
    assert not torch.allclose(base, boosted)


def test_retrieve_emits_recall_event_and_mann_bridge_metrics():
    torch.manual_seed(13)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_reasoning_controller_bridge(enabled=True, allow_shared_mann_ltm_geometry=True)
    model.diagnostics.configure(enabled=True)
    cue = torch.randn(2, 4, 16)
    context = torch.randn(2, 16)
    out = model.retrieve_memory(cue, context, strategy="direct")
    assert out.shape == (2, 16)
    event_names = [evt["event"] for evt in model.diagnostics.events]
    assert "recall_event" in event_names
    assert "reasoning_mann_bridge" in event_names
