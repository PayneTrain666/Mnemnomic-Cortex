import pytest
import torch

from mnemonic_cortex.optimizer import (
    OptimizerConfig,
    build_optimizer,
    guard_optimizer_parameter_replacement,
    migrate_optimizer_parameter_replacements,
    optimizer_references_parameters,
    unique_trainable_parameters,
)


def test_unique_trainable_parameters_deduplicates_by_identity_in_order():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    second = torch.nn.Parameter(torch.tensor([3.0]))

    result = unique_trainable_parameters([first, frozen, first, second, first])

    assert result == [first, second]


def test_flat_duplicate_parameter_is_updated_only_once():
    shared = torch.nn.Parameter(torch.tensor([1.0]))
    reference = torch.nn.Parameter(shared.detach().clone())
    cfg = OptimizerConfig(name="adam", lr=0.1)
    optimizer = build_optimizer([shared, shared], cfg)
    reference_optimizer = build_optimizer([reference], cfg)

    shared.grad = torch.tensor([0.5])
    reference.grad = shared.grad.clone()
    optimizer.step()
    reference_optimizer.step()

    assert optimizer.param_groups[0]["params"] == [shared]
    assert torch.equal(shared, reference)


def test_param_groups_preserve_options_and_skip_identical_duplicates():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    cfg = OptimizerConfig(name="adamw", lr=0.001, weight_decay=0.2)

    optimizer = build_optimizer(
        [
            {"params": [first], "lr": 0.01, "weight_decay": 0.0},
            {"params": [first, second], "lr": 0.01, "weight_decay": 0.0},
        ],
        cfg,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["params"] == [first]
    assert optimizer.param_groups[1]["params"] == [second]
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_duplicate_parameter_with_conflicting_group_options_is_rejected():
    shared = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="conflicting options"):
        build_optimizer(
            [
                {"params": [shared], "lr": 0.01},
                {"params": [shared], "lr": 0.02},
            ],
            OptimizerConfig(name="adam"),
        )


def test_structural_replacement_guard_refuses_stale_optimizer_migration():
    old = torch.nn.Parameter(torch.tensor([1.0]))
    new = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unrelated = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = build_optimizer([old], OptimizerConfig(name="adam"))

    assert optimizer_references_parameters(optimizer, [old])
    assert not optimizer_references_parameters(optimizer, [unrelated])
    with pytest.raises(RuntimeError, match="rebuild the optimizer"):
        guard_optimizer_parameter_replacement(optimizer, [old])
    with pytest.raises(RuntimeError, match="rebuild the optimizer"):
        migrate_optimizer_parameter_replacements(optimizer, [(old, new)])
