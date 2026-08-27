from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import benchmark.model_capacity as capacity
from benchmark.model_capacity import (
    SharedCapacityConfig,
    SharedGQADecoder,
    checkpoint_encoder,
)
from benchmark.models import CortexSeqModel


def _config() -> SharedCapacityConfig:
    return SharedCapacityConfig(
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        low_rank=2,
    )


def test_shared_gqa_masks_shapes_gradients_and_non_reentrant_checkpoint(monkeypatch):
    torch.manual_seed(2)
    calls = []
    real_checkpoint = capacity.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(capacity, "checkpoint", recording_checkpoint)
    decoder = SharedGQADecoder(_config(), num_layers=2).train()
    tgt = torch.randn(2, 4, 16, requires_grad=True)
    memory = torch.randn(2, 6, 16, requires_grad=True)
    tgt_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    memory_padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6]
    )

    output = decoder(
        tgt,
        memory,
        activation_checkpointing=True,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_padding,
    )
    assert output.shape == tgt.shape
    output.square().mean().backward()
    assert tgt.grad is not None
    assert memory.grad is not None
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert calls == [False, False]


def test_shared_templates_are_singletons_with_per_layer_low_rank_deltas():
    decoder = SharedGQADecoder(_config(), num_layers=3)
    names = dict(decoder.named_parameters())
    assert "self_template.q.weight" in names
    assert not any("layers.0.self_attention._template" in name for name in names)
    assert "layers.0.self_attention.deltas.left.q" in names
    assert "layers.1.self_attention.deltas.left.q" in names


def test_shared_gqa_supports_recursive_module_apply():
    decoder = SharedGQADecoder(_config(), num_layers=2)
    visited = []

    result = decoder.apply(visited.append)

    assert result is decoder
    assert decoder in visited
    assert any(
        module.__class__.__name__ == "_LowRankDeltas"
        for module in visited
    )


def test_shared_gqa_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(7)
    model = SharedGQADecoder(_config(), num_layers=2).eval()
    tgt = torch.randn(2, 3, 16)
    memory = torch.randn(2, 5, 16)
    expected = model(tgt, memory)
    checkpoint_path = tmp_path / "shared_gqa.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored = SharedGQADecoder(_config(), num_layers=2).eval()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    assert torch.equal(restored(tgt, memory), expected)


def test_context_encoder_checkpointing_preserves_masks_and_gradients(monkeypatch):
    calls = []
    real_checkpoint = capacity.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(capacity, "checkpoint", recording_checkpoint)
    layer = nn.TransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=2).train()
    value = torch.randn(2, 4, 16, requires_grad=True)
    padding = torch.tensor(
        [[False, False, False, True], [False, False, False, False]]
    )
    output = checkpoint_encoder(
        encoder,
        value,
        enabled=True,
        src_key_padding_mask=padding,
    )
    output.sum().backward()
    assert output.shape == value.shape
    assert value.grad is not None
    assert calls == [False, False]


def test_cortex_task_decoder_profile_defaults_and_safe_fallback():
    standard = CortexSeqModel(
        vocab_size=32,
        d_model=16,
        cms_enabled=False,
        task_decoder_enabled=True,
        task_decoder_layers=1,
        task_decoder_heads=4,
    )
    assert standard.task_decoder_capacity_profile == "standard"
    assert standard.task_decoder_capacity_backend == "standard_mha"
    assert isinstance(standard.task_decoder, nn.TransformerDecoder)

    fallback = CortexSeqModel(
        vocab_size=32,
        d_model=16,
        cms_enabled=False,
        task_decoder_enabled=True,
        task_decoder_layers=1,
        task_decoder_heads=4,
        task_decoder_capacity_profile="shared_gqa",
        task_decoder_kv_heads=3,
    )
    assert fallback.task_decoder_capacity_backend == "standard_mha"
    assert isinstance(fallback.task_decoder, nn.TransformerDecoder)


def test_custom_task_attention_is_excluded_from_default_cps_discovery():
    model = CortexSeqModel(
        vocab_size=32,
        d_model=16,
        cms_enabled=False,
        task_decoder_enabled=True,
        task_decoder_layers=1,
        task_decoder_heads=4,
        task_decoder_capacity_profile="shared_gqa",
        task_decoder_kv_heads=2,
        task_decoder_low_rank=2,
    )
    assert model.task_decoder_capacity_backend == "shared_gqa"
    proposal = model.enable_trainable_parameter_cps().stage(model)
    aliases = [
        alias
        for reference in proposal.references
        for alias in reference.aliases
    ]
    assert not any("task_capacity_stack" in alias for alias in aliases)
