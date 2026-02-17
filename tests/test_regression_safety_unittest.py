import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from benchmark.models import CortexSeqModel
from mnemonic_cortex.anti_hallucination import AHGThresholds, HallucinationGuard
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


class TestRegressionSafety(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(999)

    def test_train_step_stability(self):
        bsz, seq, dim, vocab = 2, 5, 24, 80
        model = CortexSeqModel(vocab_size=vocab, d_model=dim, cms_enabled=True, cms_senses=3)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for _ in range(5):
            src = torch.randint(0, vocab, (bsz, seq))
            tgt = torch.randint(0, vocab, (bsz, seq))
            logits, aux = model(src, return_aux_losses=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            if isinstance(aux.get("recall_loss", None), torch.Tensor):
                loss = loss + model.recall_loss_weight * aux["recall_loss"]
            if isinstance(aux.get("cms_loss", None), torch.Tensor):
                loss = loss + aux["cms_loss"]
            self.assertTrue(torch.isfinite(loss).item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.assertTrue(torch.isfinite(torch.tensor(float(grad_norm))).item())
            opt.step()
            if model.cortex.consolidated_lexicon is not None:
                model.cortex.consolidated_lexicon.renorm_constraints_()

    def test_checkpoint_roundtrip_retrieve_parity(self):
        bsz, seq, dim = 2, 4, 24
        model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=64, cms_senses=2)
        model.eval()
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)
        token_ids = torch.randint(0, 64, (bsz, seq))

        with torch.no_grad():
            y1 = model(
                x, ctx, operation="retrieve", token_ids=token_ids, use_consolidated_memory=True, context_features=x
            )

        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "ckpt.pt")
            model.save_checkpoint(ckpt)
            model2 = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=64, cms_senses=2)
            model2.load_checkpoint(ckpt, strict=False)
            model2.eval()
            with torch.no_grad():
                y2 = model2(
                    x,
                    ctx,
                    operation="retrieve",
                    token_ids=token_ids,
                    use_consolidated_memory=True,
                    context_features=x,
                )
        self.assertEqual(tuple(y1.shape), tuple(y2.shape))
        self.assertTrue(torch.isfinite(y2).all().item())

    def test_topology_policy_sweep(self):
        dim = 24
        model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=32, cms_senses=2)
        model.topology.register_babi_qhm_v2_policy()
        x = torch.randn(2, 3, dim)
        ctx = torch.randn(2, dim)
        ids = torch.randint(0, 32, (2, 3))
        for name in model.topology.list_policies():
            model.apply_topology_policy(name)
            y = model(
                x, ctx, operation="retrieve", token_ids=ids, use_consolidated_memory=True, context_features=x
            )
            self.assertEqual(tuple(y.shape), (2, dim))
            self.assertTrue(torch.isfinite(y).all().item())

    def test_cms_stress_logging_and_shards(self):
        bsz, seq, dim, vocab = 3, 16, 24, 512
        with tempfile.TemporaryDirectory() as td:
            model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=vocab, cms_senses=3)
            model.enable_cms_logger(out_dir=td, max_buffer=8, sample_rate=0.5)
            x = torch.randn(bsz, seq, dim)
            ctx = torch.randn(bsz, dim)
            ids = torch.randint(0, vocab, (bsz, seq))
            _ = model(
                x, ctx, operation="process", token_ids=ids, use_consolidated_memory=True, context_features=x
            )
            _ = model.flush_cms_logger()
            keys = model.save_cms_shards(os.path.join(td, "shards"), shard_size=64, quantize=True)
            self.assertTrue(len(keys) > 0)
            model.load_cms_shards(os.path.join(td, "shards"))

    def test_ahg_contract_paths(self):
        def gen_ok(prompt, **kw):
            logits = torch.tensor([[[5.0, 0.5], [4.0, 0.4]]], dtype=torch.float32)
            return {"out_text": "ok answer", "logits": logits, "tokens": torch.tensor([[1, 1]])}

        guard_ok = HallucinationGuard(
            thresholds=AHGThresholds(min_retrieval_score=0.0, min_coverage=0.0),
            generate_fn=gen_ok,
            retriever_fn=lambda q, k=5: [{"text": "evidence", "score": 0.9, "source_id": "s1"}],
            cite_align_fn=lambda a, d: 1.0,
            cms_signals_fn=lambda t: {"sense_entropy": 0.2, "proto_distance": 0.1, "warp": 1.0},
        )
        out_ok = guard_ok.answer("q")
        self.assertIn(out_ok["mode"], {"answer", "grounded"})

        def gen_bad(prompt, **kw):
            logits = torch.zeros(1, 2, 5)
            return {"out_text": "maybe", "logits": logits, "tokens": torch.tensor([[1, 2]])}

        guard_bad = HallucinationGuard(
            thresholds=AHGThresholds(),
            generate_fn=gen_bad,
            retriever_fn=None,
            cite_align_fn=None,
            cms_signals_fn=lambda t: {"sense_entropy": 2.0, "proto_distance": 0.95, "warp": 1.0},
        )
        out_bad = guard_bad.answer("q")
        self.assertIn(out_bad["mode"], {"ask", "refuse"})

    def test_triple_hybrid_write_read_cycle(self):
        bsz, seq, dim = 2, 4, 16
        mem = EnhancedTripleHybridMemory(
            input_dim=dim, output_dim=dim, hg_slots=64, cgmn_slots=48, curved_slots=32
        )
        x = torch.randn(bsz, seq, dim)
        for _ in range(3):
            _ = mem(x, operation="write")
            y = mem(x, operation="read")
            self.assertEqual(tuple(y.shape), (bsz, seq, dim))
            self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
