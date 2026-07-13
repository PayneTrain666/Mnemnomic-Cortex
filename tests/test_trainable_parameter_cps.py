"""Focused tests for trainable central parameter storage."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from mnemonic_cortex.trainable_parameter_cps import (
    CPSBackedEmbedding,
    CPSBackedLinear,
    CPSBackedMultiheadAttention,
    TrainableParameterCPS,
    TrainableParameterCPSConfig,
)


class _MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 4)
        self.embedding = nn.Embedding(11, 6)
        self.attention = nn.MultiheadAttention(6, 2, batch_first=True, dropout=0.0)


def test_exact_materialization_gradients_and_outputs() -> None:
    torch.manual_seed(3)
    model = _MixedModel()
    x = torch.randn(2, 5, 6)
    indices = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected_linear = model.linear(x)
    expected_embedding = model.embedding(indices)
    expected_attention = model.attention(x, x, x)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    cps = TrainableParameterCPS()
    proposal = cps.stage(model)
    commit = cps.commit(proposal)

    assert commit.committed
    assert isinstance(model.linear, CPSBackedLinear)
    assert isinstance(model.embedding, CPSBackedEmbedding)
    assert isinstance(model.attention, CPSBackedMultiheadAttention)
    assert torch.equal(model.linear(x), expected_linear)
    assert torch.equal(model.embedding(indices), expected_embedding)
    actual_attention = model.attention(x, x, x)
    assert torch.equal(actual_attention[0], expected_attention[0])
    assert torch.equal(actual_attention[1], expected_attention[1])

    for ref in cps.registry.values():
        assert torch.equal(cps.materialize(ref.handle), original[ref.aliases[0]])

    loss = (
        model.linear(x).sum()
        + model.embedding(indices).sum()
        + model.attention(x, x, x, need_weights=False)[0].sum()
    )
    loss.backward()
    assert list(model.linear.parameters()) == []
    assert list(model.embedding.parameters()) == []
    assert all(parameter.grad is not None for parameter in cps.slabs)
    assert set(model.parameters()) == set(cps.parameters())


def test_true_tied_parameter_is_deduplicated_and_preserved() -> None:
    model = nn.Module()
    model.left = nn.Linear(5, 5, bias=False)
    model.right = nn.Linear(5, 5, bias=False)
    model.right.weight = model.left.weight
    x = torch.randn(3, 5)
    expected_left = model.left(x)
    expected_right = model.right(x)

    cps = TrainableParameterCPS()
    proposal = cps.stage(model)
    assert len(proposal.references) == 1
    assert len(proposal.references[0].aliases) == 2
    cps.commit()

    assert model.left.weight.data_ptr() == model.right.weight.data_ptr()
    assert torch.equal(model.left(x), expected_left)
    assert torch.equal(model.right(x), expected_right)
    report = cps.capacity_report()
    assert report["unique_parameter_handles"] == 1
    assert report["tied_alias_count"] == 1
    assert report["stored_scalar_count"] == 25


def _low_rank_pair() -> nn.Module:
    torch.manual_seed(8)
    model = nn.Module()
    model.first = nn.Linear(8, 8, bias=False)
    model.second = nn.Linear(8, 8, bias=False)
    template = torch.randn(8, 8)
    left = torch.randn(8, 1)
    right = torch.randn(1, 8)
    with torch.no_grad():
        model.first.weight.copy_(template + left @ right)
        model.second.weight.copy_(template - left @ right)
    return model


def test_shared_template_low_rank_compression_has_real_savings() -> None:
    model = _low_rank_pair()
    expected = [model.first.weight.detach().clone(), model.second.weight.detach().clone()]
    cps = TrainableParameterCPS(
        TrainableParameterCPSConfig(
            enable_compression=True,
            max_rank=1,
            reconstruction_tolerance=1.0e-5,
            output_tolerance=1.0e-5,
        )
    )
    cps.stage(model)
    evaluation = cps.evaluate()
    assert evaluation.compression_applied
    assert evaluation.scalar_savings > 0
    cps.commit()
    assert cps.capacity_report()["scalar_savings"] == 0
    evaluation = cps.compress_committed()
    assert evaluation.compression_applied

    assert torch.allclose(model.first.weight, expected[0], atol=1.0e-5, rtol=0)
    assert torch.allclose(model.second.weight, expected[1], atol=1.0e-5, rtol=0)
    report = cps.capacity_report()
    assert report["stored_scalar_count"] == 64 + 2 * (8 + 8)
    assert report["scalar_savings"] == 32
    (model.first(torch.randn(2, 8)).sum() + model.second(torch.randn(2, 8)).sum()).backward()
    assert all(parameter.grad is not None for parameter in cps.slabs)


def test_failed_compression_tolerance_stays_exact() -> None:
    torch.manual_seed(11)
    model = nn.Module()
    model.first = nn.Linear(6, 6, bias=False)
    model.second = nn.Linear(6, 6, bias=False)
    originals = [model.first.weight.detach().clone(), model.second.weight.detach().clone()]
    cps = TrainableParameterCPS(
        TrainableParameterCPSConfig(
            enable_compression=True,
            max_rank=1,
            reconstruction_tolerance=0.0,
            output_tolerance=0.0,
        )
    )
    cps.stage(model)
    assert not cps.evaluate().compression_applied
    cps.commit()
    evaluation = cps.compress_committed()
    assert not evaluation.compression_applied
    assert torch.equal(model.first.weight, originals[0])
    assert torch.equal(model.second.weight, originals[1])
    assert cps.capacity_report()["scalar_savings"] == 0
    assert all(ref.storage_kind == "exact" for ref in cps.registry.values())


def test_commit_rollback_idempotency_optimizer_rejection_and_manifest() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    expected = model(x)
    original_first = model[0]
    cps = TrainableParameterCPS()
    proposal = cps.stage(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer migration"):
        cps.commit(proposal, optimizer=optimizer)

    first_commit = cps.commit(proposal)
    assert cps.commit(proposal) is first_commit
    assert torch.equal(model(x), expected)
    json.dumps(cps.to_manifest())
    first_rollback = cps.rollback()
    assert first_rollback is not None and first_rollback.rolled_back
    assert cps.rollback() is first_rollback
    assert model[0] is original_first
    assert torch.equal(model(x), expected)
    assert not hasattr(model, "_trainable_parameter_cps")


def test_rollback_preserves_training_updates_from_canonical_store() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    cps = TrainableParameterCPS()
    cps.commit(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[0].weight.detach().clone()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    trained = model[0].weight.detach().clone()
    assert not torch.equal(before, trained)

    cps.rollback()
    assert isinstance(model[0], nn.Linear)
    assert torch.equal(model[0].weight, trained)


def test_compressed_manifest_rebuilds_layout_before_state_load() -> None:
    model = _low_rank_pair()
    cps = TrainableParameterCPS(
        TrainableParameterCPSConfig(
            enable_compression=True,
            max_rank=1,
            reconstruction_tolerance=1.0e-5,
        )
    )
    cps.commit(cps.stage(model))
    cps.compress_committed()
    manifest = cps.to_manifest()
    state = cps.state_dict()
    x = torch.randn(2, 8)
    expected = model.first(x)

    restored_model = _low_rank_pair()
    restored = TrainableParameterCPS(cps.config)
    restored.prepare_from_manifest(restored_model, manifest)
    restored.load_state_dict(state)
    assert torch.allclose(restored_model.first(x), expected, atol=1.0e-6)


def test_capacity_is_literal_and_discovery_rejections_are_explained() -> None:
    class Unsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Parameter(torch.ones(3))

    model = nn.Module()
    model.good = nn.Linear(3, 2)
    model.frozen = nn.Linear(3, 2)
    model.frozen.weight.requires_grad_(False)
    model.unsupported = Unsupported()
    cps = TrainableParameterCPS()
    proposal = cps.stage(model)
    assert proposal.rejected["frozen"] == "frozen parameters are unsupported"
    assert proposal.rejected["unsupported"] == "unsupported parameterized module type"
    report = cps.capacity_report()
    assert report["literal"] is True
    assert report["original_scalar_count"] == 8
    assert report["stored_scalar_count"] == 8
    assert report["scalar_savings"] == 0
