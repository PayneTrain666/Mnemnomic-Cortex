import unittest

import torch

from mnemonic_cortex.consolidation_broker import ConsolidationBroker
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.topology_manager_v2 import TopologyManagerV2


class TestConsolidationAndTopologySafety(unittest.TestCase):
    def test_broker_alias_route_and_read(self):
        broker = ConsolidationBroker(vocab_size=64, model_dim=24)
        token_ids = torch.randint(0, 64, (32,), dtype=torch.long)
        base = torch.randn(32, 24)
        ctx = torch.randn(32, 24)
        fused, aux = broker.route_fuse(token_ids, base, ctx, intent="causal")
        self.assertEqual(tuple(fused.shape), (32, 24))
        self.assertIn("selected_store", aux)
        # "causal" routes to logical CGS alias, backed by SKS physical store.
        rd = broker.route_read(torch.randn(4, 24), intent="causal", k=4)
        self.assertEqual(rd["store"], "CGS")
        self.assertEqual(rd["physical_store"], "SKS")
        self.assertIn("proto_distance", rd["signals"])

    def test_unified_multimemory_write_updates(self):
        broker = ConsolidationBroker(vocab_size=64, model_dim=24)
        items_a = torch.randn(12, 24)
        items_b = torch.randn(8, 24)
        metas_a = [{"domain": "science", "sources": ["a"], "tags": ["t1"], "time": 10}] * 12
        metas_b = [{"domain": "science", "sources": ["b"], "tags": ["t2"], "time": 7}] * 8

        broker.unify_and_route_write(items_a, metas_a, domain="science")
        m1 = broker.get_metrics()["SKS"]
        self.assertGreaterEqual(m1["unified_n"], 12.0)
        self.assertGreaterEqual(m1["unified_updates"], 1.0)

        broker.unify_and_route_write(items_b, metas_b, domain="science")
        m2 = broker.get_metrics()["SKS"]
        self.assertGreaterEqual(m2["unified_n"], 20.0)
        self.assertGreaterEqual(m2["unified_updates"], 2.0)

    def test_auto_intent_hits_all_physical_stores(self):
        broker = ConsolidationBroker(vocab_size=64, model_dim=24)
        token_ids = torch.randint(0, 64, (16,), dtype=torch.long)
        base = torch.randn(16, 24)
        ctx = torch.randn(16, 24)
        fused, aux = broker.route_fuse(token_ids, base, ctx, intent="auto")
        self.assertEqual(tuple(fused.shape), (16, 24))
        metrics = broker.get_metrics()
        for store in ("CRS", "SKS", "CAS", "PSS"):
            self.assertGreaterEqual(metrics[store]["hits"], 1.0, store)
        self.assertEqual(set(aux["stores"].keys()), {"CRS", "SKS", "CAS", "PSS"})

    def test_cms_broker_unify_from_ltm_updates_all_domains(self):
        cortex = EnhancedMnemonicCortex(input_dim=24, output_dim=24, cms_vocab_size=64)
        cortex.enable_consolidation_broker(vocab_size=64)
        written = cortex._cms_broker_unify_from_ltm(max_items_per_bank=8)
        self.assertTrue(written)
        metrics = cortex.consolidation_broker.get_metrics()
        self.assertGreaterEqual(metrics["CRS"]["unified_updates"], 1.0)
        self.assertGreaterEqual(metrics["SKS"]["unified_updates"], 1.0)
        self.assertGreaterEqual(metrics["CAS"]["unified_updates"], 1.0)
        self.assertGreaterEqual(metrics["PSS"]["unified_updates"], 1.0)

    def test_topology_v2_clamp_and_curved_mutation(self):
        model = EnhancedMnemonicCortex(input_dim=24, output_dim=24)
        top = TopologyManagerV2(default_policy="safe_test")
        top.register_policy(
            "safe_test",
            curvature_mode="euclidean",
            curvature_rate=5.0,
            curvature_clip=0.25,
        )
        top.activate_policy("safe_test", model=model)

        wm_before = model.working_memory.curvature.detach().clone()
        cv_before = model.long_term_memory.curved.memory_curvature.detach().clone()
        top.step(model, loss_value=10.0)
        wm_after = model.working_memory.curvature.detach().clone()
        cv_after = model.long_term_memory.curved.memory_curvature.detach().clone()

        self.assertFalse(torch.allclose(wm_before, wm_after))
        self.assertFalse(torch.allclose(cv_before, cv_after))
        self.assertLessEqual(float(wm_after.abs().max().item()), 0.251)
        self.assertLessEqual(float(cv_after.abs().max().item()), 0.251)


if __name__ == "__main__":
    unittest.main()

