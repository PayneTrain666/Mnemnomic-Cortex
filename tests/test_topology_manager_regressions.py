import torch
import pytest

from mnemonic_cortex.topology_manager import TopologyManagerV3, colored_noise_1f
from mnemonic_cortex.topology_manager_v2 import TopologyManagerV2


def test_colored_noise_1f_returns_finite_vector():
    z = colored_noise_1f(17, device="cpu", dtype=torch.float32, alpha=1.0)
    assert z.shape == (17,)
    assert torch.isfinite(z).all()
    assert float(z.abs().sum().item()) > 0.0


def test_topology_v3_fractal_mutation_path_no_crash():
    m = TopologyManagerV3(subsystems=("hg",), mutation_rate=0.01)
    m._topo["hg"] = "fractal"
    curv = torch.zeros(12, dtype=torch.float32)
    out = m.mutate_curvature(curvature=curv, loss_value=5.0, subsystem="hg")
    assert out.shape == curv.shape
    assert torch.isfinite(out).all()
    assert float(out.abs().max().item()) <= float(m.curv_clamp) + 1e-6


def test_topology_v2_policy_allows_partial_adaptive_overrides():
    t = TopologyManagerV2(default_policy="default")
    t.register_policy("partial_adaptive", adaptive={"fit_up": 0.9})
    t.activate_policy("partial_adaptive")
    p = t.current_policy()
    assert p["adaptive"]["fit_up"] == 0.9
    assert "fit_down" in p["adaptive"]
    assert "step" in p["adaptive"]


def test_topology_v2_rejects_invalid_curvature_mix():
    t = TopologyManagerV2(default_policy="default")
    with pytest.raises(ValueError):
        t.register_policy("bad_mix_len", curvature_mix=(0.5, 0.5, 0.0))
    with pytest.raises(ValueError):
        t.register_policy("bad_mix_neg", curvature_mix=(0.5, -0.1, 0.3, 0.3))
