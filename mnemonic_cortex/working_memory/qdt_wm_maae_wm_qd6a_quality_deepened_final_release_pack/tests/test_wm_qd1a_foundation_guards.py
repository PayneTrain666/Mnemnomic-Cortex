import json
import torch
import pytest

from mnemonic_cortex.working_memory import (
    WMFoundationValidationError,
    ensure_finite_tensor,
    ensure_rank,
    ensure_last_dim,
    ensure_probability_vector,
    clamp_norm,
    safe_jsonable,
    foundation_trace,
    row_stochastic,
)


def test_foundation_guards_accept_good_tensor_and_reject_nonfinite():
    x = torch.randn(2, 3, 4)
    assert ensure_finite_tensor("x", x) is x
    assert ensure_rank("x", x, 3) is x
    assert ensure_last_dim("x", x, 4) is x

    bad = x.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(WMFoundationValidationError):
        ensure_finite_tensor("bad", bad)


def test_probability_and_row_stochastic_guards():
    probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    ensure_probability_vector("probs", probs)

    raw = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    normed = row_stochastic(raw)
    assert torch.allclose(normed.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(normed).all()


def test_clamp_norm_and_safe_jsonable_are_serialization_safe():
    x = torch.ones(2, 4) * 1e8
    y = clamp_norm(x, max_norm=10.0)
    assert torch.all(y.norm(dim=-1) <= 10.0 + 1e-4)

    payload = safe_jsonable({"x": torch.randn(2, 3), "large": torch.randn(100)})
    json.dumps(payload)
    assert payload["large"]["tensor_shape"] == [100]


def test_foundation_trace_contains_paamax_metadata():
    trace = foundation_trace(trace_type="unit", module="m", message="ok", payload={"x": torch.tensor([1.0])})
    json.dumps(trace)
    assert trace["paamax_metadata"]["trace_governance"] is True
    assert trace["paamax_metadata"]["write_permission_required"] is False
