import unittest

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.cps import ConsolidatedParamStore
from mnemonic_cortex.cps_fuser import CPSFuser, FuserCfg


class TestCPSExtensions(unittest.TestCase):
    def test_store_snapshot_and_restore(self):
        cps = ConsolidatedParamStore()
        up = cps.ensure("token:42")
        before = up.euclid.detach().clone()
        snap = cps.snapshot()
        with torch.no_grad():
            up.euclid.add_(1.0)
        self.assertFalse(torch.allclose(before, up.euclid))
        cps.restore(snap, strict=True)
        self.assertTrue(torch.allclose(before, up.euclid))

    def test_confidence_signals(self):
        cps = ConsolidatedParamStore()
        cps.ensure("token:1")
        sig = cps.confidence_signals_for_keys(["token:1", "token:999"])
        self.assertIn("fisher_uncertainty", sig)
        self.assertIn("phase_agreement", sig)
        self.assertGreaterEqual(sig["phase_agreement"], 0.0)
        self.assertLessEqual(sig["phase_agreement"], 1.0)

    def test_curriculum_stage_switch(self):
        fuser = CPSFuser(FuserCfg(d_model=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        fuser.set_curriculum_stage(0)
        self.assertEqual(fuser.cfg.use_heads, ("S",))
        fuser.set_curriculum_stage(2)
        self.assertEqual(fuser.cfg.use_heads, ("H", "S", "P"))
        fuser.set_curriculum_stage(3)
        self.assertEqual(fuser.cfg.use_heads, ("H", "S", "P", "F"))

    def test_cortex_cps_helpers(self):
        model = EnhancedMnemonicCortex(
            input_dim=24,
            output_dim=8,
            sensory_buffer_size=4,
            wm_slots=5,
            wm_slot_dim=32,
            ltm_hg_slots=64,
            ltm_cgmn_slots=64,
            ltm_curved_slots=64,
        )
        model.set_cps_curriculum_stage(1)
        embs = model.cps_embed_tokens(["7", "8", "9"])
        self.assertEqual(list(embs.shape), [3, 24])
        reg = model.cps_agreement_loss(["7", "8", "9"])
        self.assertTrue(torch.is_tensor(reg))
        self.assertEqual(reg.dim(), 0)


if __name__ == "__main__":
    unittest.main()
