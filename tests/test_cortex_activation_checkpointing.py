from __future__ import annotations

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_cortex_activation_checkpointing_is_opt_in_and_non_reentrant(monkeypatch):
    calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return fn(*args)

    monkeypatch.setattr(
        "mnemonic_cortex.cortex.activation_checkpoint",
        fake_checkpoint,
    )
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        wm_slots=8,
        enable_activation_checkpointing=True,
        enable_global_hidden_attention=False,
        enable_secondary_hidden_stack=False,
    ).train()
    x = torch.randn(2, 3, 16)
    out = model.process_sensory_input(x)
    assert out.shape == x.shape
    assert calls == [False]

    model.enable_activation_checkpointing = False
    calls.clear()
    model.process_sensory_input(x)
    assert calls == []
