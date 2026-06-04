import torch
import pytest

from mnemonic_cortex.router_advanced import AdvancedDomainRouter
from mnemonic_cortex.router_losses import router_regularizer


def test_router_regularizer_accepts_1d_and_logits_like_values():
    logits_like = torch.tensor([1.0, -1.5, 0.2, 3.1], dtype=torch.float32)
    reg, aux = router_regularizer(logits_like, top_k=2)
    assert torch.is_tensor(reg)
    assert reg.dim() == 0
    assert torch.isfinite(reg)
    assert "entropy" in aux
    assert "sparsity_mass" in aux


def test_advanced_router_rejects_invalid_query_dim():
    router = AdvancedDomainRouter(["core", "science"], d_in=8)
    with pytest.raises(ValueError):
        _ = router(torch.randn(2, 3, 8))


def test_advanced_router_handles_non_finite_temperature_safely():
    router = AdvancedDomainRouter(["core", "science", "reasoning"], d_in=8)
    router.set_temperature(float("nan"))
    idx, w, probs = router(torch.randn(8), top_k=2)
    assert len(idx) == 2
    assert len(w) == 2
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum().item()) - 1.0) < 1e-5
