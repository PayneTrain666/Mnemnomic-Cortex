import torch

from mnemonic_cortex.working_memory import WMCurvedAssociativeCore, EnhancedCurvedMemory


class ExternalCanonicalCurved(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_mode = False
        self.last_trace = {"external": True}

    def enable_energy_efficient_mode(self, enable=True):
        self.energy_mode = bool(enable)

    def forward(self, x, operation="read", importance=None):
        if operation == "write":
            return x
        return x + 2.0


def test_canonical_compatible_curved_memory_read_write_process_shapes():
    mem = EnhancedCurvedMemory(input_dim=32, hidden_dim=64, mem_slots=5)
    x = torch.randn(2, 4, 32)

    y = mem(x, operation="read")
    assert y.shape == x.shape

    p = mem(x, operation="process")
    assert p.shape == x.shape

    w = mem(x, operation="write")
    assert w.shape == x.shape


def test_canonical_compatible_curved_memory_preserves_core_attributes():
    mem = EnhancedCurvedMemory(input_dim=32, hidden_dim=64, curvature_dim=8, mem_slots=5)
    assert hasattr(mem, "encoder")
    assert hasattr(mem, "curvature")
    assert hasattr(mem, "memory_slots")
    assert hasattr(mem, "memory_importance")
    assert hasattr(mem, "associative_weights")
    assert hasattr(mem, "output_projection")


def test_curved_wrapper_uses_internal_canonical_when_no_external_module():
    core = WMCurvedAssociativeCore(input_dim=32, hidden_dim=64, mem_slots=5)
    x = torch.randn(2, 4, 32)
    y, trace = core(x, operation="read", return_trace=True)
    assert y.shape == x.shape
    assert isinstance(trace, dict)
    assert "top_indices" in trace
    assert core.last_trace is not None


def test_curved_wrapper_delegates_to_external_canonical_module():
    external = ExternalCanonicalCurved()
    core = WMCurvedAssociativeCore(input_dim=32, hidden_dim=64, mem_slots=5, curved_memory=external)
    x = torch.randn(2, 4, 32)

    y = core(x, operation="read")
    assert torch.allclose(y, x + 2.0)

    core.enable_energy_efficient_mode(True)
    assert external.energy_mode is True


def test_curved_wrapper_rejects_bad_shape():
    core = WMCurvedAssociativeCore(input_dim=32, hidden_dim=64, mem_slots=5)
    bad = torch.randn(2, 32)
    try:
        core(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for bad shape")
