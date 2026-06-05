import unittest
import torch

from mnemonic_cortex.memory_hg import EnhancedHyperGeometricMemory as HG
from mnemonic_cortex.memory_cgmn import EnhancedCGMNMemory as CGMN
from mnemonic_cortex.memory_curved import EnhancedCurvedMemory as Curved


class TestMemoryBehaviour(unittest.TestCase):
    def test_forward_shapes(self):
        torch.manual_seed(0)
        B, S, D = 4, 6, 32
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            x = torch.randn(B, S, D, device=device)
            hg = HG(input_dim=D, mem_slots=64, holo_dim=128).to(device)
            cg = CGMN(input_dim=D, mem_slots=64, slot_dim=64).to(device)
            cv = Curved(input_dim=D, hidden_dim=64, mem_slots=64).to(device)
            for m in [hg, cg, cv]:
                out = m(x, operation="read")
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(out.dtype, x.dtype)
                self.assertEqual(out.device.type, device)

    def test_holo_rms_clamp(self):
        torch.manual_seed(0)
        B, S, D = 2, 3, 16
        x = torch.randn(B, S, D)
        hg = HG(input_dim=D, mem_slots=32, holo_dim=64)
        with torch.no_grad():
            hg.holograms_fft.uniform_(0, 20.0)  # inflate energy
            pre_rms = hg.holograms_fft.abs().mean(dim=-1).clone()
        idx = torch.zeros(B, S, 1, dtype=torch.long)
        w = torch.ones(B, S, 1)
        hg._holo_write(x, idx, w)
        rms = hg.holograms_fft.abs().mean(dim=-1)
        self.assertTrue(torch.isfinite(rms).all(), "Spectral energy became non-finite")
        self.assertTrue(torch.all(rms <= pre_rms + 1e-6), "Spectral energy unexpectedly increased")

    def test_ddp_sync_noop(self):
        """Ensure _sync_buffers_ddp is a no-op without distributed init."""
        hg = HG(input_dim=8, mem_slots=8)
        before = hg.usage_counts.clone()
        hg._sync_buffers_ddp()
        torch.testing.assert_close(before, hg.usage_counts)


if __name__ == "__main__":
    unittest.main()
