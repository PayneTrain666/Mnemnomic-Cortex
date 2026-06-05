import unittest

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex


class TestInterMemoryAttention(unittest.TestCase):
    def test_inter_memory_bridges_shape_and_metrics(self):
        model = EnhancedMnemonicCortex(
            input_dim=24,
            output_dim=12,
            sensory_buffer_size=4,
            wm_slots=6,
            wm_slot_dim=32,
            ltm_hg_slots=64,
            ltm_cgmn_slots=64,
            ltm_curved_slots=64,
        )
        model.enable_diagnostics(enabled=True)
        x = torch.randn(2, 5, 24)
        ctx = torch.randn(2, 24)

        y = model(x, ctx, operation="process")
        self.assertEqual(list(y.shape), [2, 5, 24])

        r = model(x, ctx, operation="retrieve")
        self.assertEqual(list(r.shape), [2, 24])

        metrics = model.get_metrics()
        self.assertIn("ltm_inter_gate", metrics)
        self.assertIn("diag_ema_bridge_gate_process", metrics)
        self.assertIn("diag_ema_bridge_gate_retrieve", metrics)


if __name__ == "__main__":
    unittest.main()
