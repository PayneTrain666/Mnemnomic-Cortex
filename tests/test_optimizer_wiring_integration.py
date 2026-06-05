import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.optimizer import (
    MemoryOptimizer,
    OptimizerConfig,
    build_optimizer,
    build_warmup_cosine_scheduler,
)


def test_build_optimizer_and_scheduler_step():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    cfg = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01)
    opt = build_optimizer(model.parameters(), cfg)
    sched = build_warmup_cosine_scheduler(
        opt,
        total_steps=20,
        warmup_ratio=0.2,
        min_lr_ratio=0.1,
    )
    assert opt.__class__.__name__ == "AdamW"
    x = torch.randn(2, 4, 16)
    ctx = torch.randn(2, 16)
    out = model(x, ctx, operation="retrieve")
    loss = out.pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    assert float(opt.param_groups[0]["lr"]) > 0.0


def test_memory_optimizer_profiles_and_can_toggle_energy_mode():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    profiler = MemoryOptimizer(model)
    batches = [
        (torch.randn(2, 4, 16), torch.randn(2, 16)),
        (torch.randn(2, 4, 16), torch.randn(2, 16)),
    ]
    metrics = profiler.profile(batches)
    assert set(metrics.keys()) == {"access_time", "energy_usage", "mse_proxy"}
    assert metrics["access_time"] >= 0.0
    assert metrics["energy_usage"] > 0.0
    # Use tiny threshold to force enable path.
    enabled = profiler.enable_energy_mode_if_slow(max_time=0.0)
    assert enabled is True
