import torch

from mnemonic_cortex.working_memory import (
    CurvedResonanceConfig,
    CurvedResonantWMCore,
    WMCurvedAssociativeCore,
)


def test_curved_resonant_core_read_shape_and_trace():
    cfg = CurvedResonanceConfig(
        input_dim=32,
        hidden_dim=64,
        resonance_slots=8,
        requested_resonance_steps=2,
        max_resonance_steps=3,
    )
    core = CurvedResonantWMCore(cfg)
    x = torch.randn(2, 5, 32)
    y, trace = core(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    assert trace["operation"] == "read"
    assert trace["resonance_steps_executed"] == 2
    assert len(trace["step_traces"]) == 2
    assert "paamax_metadata" in trace


def test_curved_resonant_core_bounds_requested_steps():
    cfg = CurvedResonanceConfig(
        input_dim=32,
        hidden_dim=64,
        resonance_slots=8,
        requested_resonance_steps=10,
        max_resonance_steps=3,
    )
    core = CurvedResonantWMCore(cfg)
    x = torch.randn(2, 5, 32)
    _, trace = core(x, operation="read", return_trace=True)

    assert trace["resonance_steps_requested"] == 10
    assert trace["resonance_steps_executed"] == 3
    assert trace["bounded"] is False


def test_curved_resonant_core_preserves_inner_core():
    inner = WMCurvedAssociativeCore(input_dim=32, hidden_dim=64, mem_slots=5)
    cfg = CurvedResonanceConfig(input_dim=32, hidden_dim=64, resonance_slots=8)
    core = CurvedResonantWMCore(cfg, inner_core=inner)

    assert core.inner_core is inner

    x = torch.randn(2, 5, 32)
    y = core(x, operation="process")
    assert y.shape == x.shape


def test_curved_resonant_write_delegates_to_inner_core_and_traces():
    cfg = CurvedResonanceConfig(input_dim=32, hidden_dim=64, resonance_slots=8)
    core = CurvedResonantWMCore(cfg)
    x = torch.randn(2, 5, 32)

    y, trace = core(x, operation="write", return_trace=True)
    assert y.shape == x.shape
    assert trace["operation"] == "write"
    assert trace["resonance_steps_executed"] == 0
    assert trace["paamax_metadata"]["write_permission_required"] is True


def test_curved_resonant_core_rejects_bad_shape():
    cfg = CurvedResonanceConfig(input_dim=32, hidden_dim=64, resonance_slots=8)
    core = CurvedResonantWMCore(cfg)
    bad = torch.randn(2, 32)
    try:
        core(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
