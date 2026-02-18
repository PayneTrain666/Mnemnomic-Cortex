import unittest

import torch

from mnemonic_cortex.consolidated_memory import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from mnemonic_cortex.consolidation_broker_v2 import BrokerCfg, ConsolidationBrokerV2
from mnemonic_cortex.cps import ConsolidatedParamStore, UnifiedParamCfg
from mnemonic_cortex.cps_fuser import CPSFuser, FuserCfg
from mnemonic_cortex.multi_cps import MultiCPSManager
from mnemonic_cortex.router_advanced import AdvancedDomainRouter
from mnemonic_cortex.router_losses import router_regularizer
from mnemonic_cortex.quantization import CPSQuantizer
from mnemonic_cortex.quant_fuser import QuantAwareCPSFuser


class TestAdvancedConsolidationStack(unittest.TestCase):
    def test_broker_ingest_and_cohesion(self):
        cms = ConsolidatedMemoryStore(ConsolidatedMemoryCfg(d_model=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        cps = ConsolidatedParamStore(UnifiedParamCfg(d_euclid=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        fuser = CPSFuser(FuserCfg(d_model=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        broker = ConsolidationBrokerV2(cms=cms, cps=cps, cps_fuser=fuser, cfg=BrokerCfg())
        key = "core:test"
        cand = {
            "E": torch.randn(24),
            "H": torch.randn(8),
            "S": torch.randn(8),
            "F": (torch.randn(8), torch.randn(8)),
            "T": torch.randn(2),
            "P": (torch.rand(4), torch.randn(4)),
        }
        broker.ingest_from_ltm(key, cand, importance=torch.tensor([0.9]), src_info={"src": "unit"})
        self.assertIn(key, cms.keys())
        broker.cms_pull_to_cps(key)
        broker.cps_push_to_cms(key)
        coh = broker.cohesion_regularizer([key])
        self.assertTrue(torch.is_tensor(coh))
        self.assertEqual(coh.dim(), 0)

    def test_router_quant_and_multi_cps(self):
        mgr = MultiCPSManager()
        core = ConsolidatedParamStore(UnifiedParamCfg(d_euclid=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        sci = ConsolidatedParamStore(UnifiedParamCfg(d_euclid=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        ff_core = CPSFuser(FuserCfg(d_model=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        ff_sci = CPSFuser(FuserCfg(d_model=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
        mgr.register("core", core, ff_core).register("science", sci, ff_sci)
        up = mgr.ensure("science:ATP")
        self.assertIsNotNone(up)

        router = AdvancedDomainRouter(["core", "science"], d_in=24)
        idx, w, probs = router(torch.randn(24), key="science:ATP", top_k=2)
        self.assertEqual(len(idx), 2)
        reg, aux = router_regularizer(probs.unsqueeze(0), top_k=2)
        self.assertTrue(torch.is_tensor(reg))
        self.assertIn("entropy", aux)

        q = CPSQuantizer()
        qpack = q.quantize_entry(core.ensure("core:token"))
        qf = QuantAwareCPSFuser(
            d_out=24,
            dims={"E": 24, "H": 8, "S": 8, "F": 8, "T": 2, "P": 4},
            quantizer=q,
        )
        fused, _ = qf(qpack=qpack)
        self.assertEqual(list(fused.shape), [24])


if __name__ == "__main__":
    unittest.main()
