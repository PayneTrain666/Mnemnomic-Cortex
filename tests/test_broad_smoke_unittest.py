import glob
import os
import tempfile
import unittest

import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


class TestBroadSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_end_to_end_cortex_broad_smoke(self):
        bsz, seq, dim, vocab = 2, 4, 24, 96
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)
        token_ids = torch.randint(0, vocab, (bsz, seq))

        with tempfile.TemporaryDirectory() as td:
            model = EnhancedMnemonicCortex(
                input_dim=dim,
                output_dim=dim,
                sensory_buffer_size=3,
                wm_slots=5,
                wm_slot_dim=32,
                ltm_hg_slots=64,
                ltm_cgmn_slots=48,
                ltm_curved_slots=32,
                cms_vocab_size=vocab,
                cms_senses=3,
            )
            model.enable_cms_logger(out_dir=td, max_buffer=2, sample_rate=1.0)
            model.topology.register_babi_qhm_v2_policy()
            model.apply_topology_policy("babi_qhm_v2")

            # Core flow smoke: process, retrieve, consolidate.
            y_proc, aux = model(
                x,
                ctx,
                operation="process",
                return_aux_losses=True,
                token_ids=token_ids,
                use_consolidated_memory=True,
                context_features=x,
            )
            self.assertEqual(tuple(y_proc.shape), (bsz, seq, dim))
            self.assertIn("recall_loss", aux)

            y_ret = model(
                x,
                ctx,
                operation="retrieve",
                token_ids=token_ids,
                use_consolidated_memory=True,
                context_features=x,
            )
            self.assertEqual(tuple(y_ret.shape), (bsz, dim))

            _ = model(x, ctx, operation="consolidate")

            # One optimizer step + topology step.
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss = y_proc.mean() + y_ret.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.topology_step(float(loss.detach().cpu()))
            if model.consolidated_lexicon is not None:
                model.consolidated_lexicon.renorm_constraints_()

            # Logger should flush records.
            flushed = model.flush_cms_logger()
            self.assertTrue(flushed is None or flushed.endswith(".pt"))
            files = glob.glob(os.path.join(td, "cms_records_*.pt"))
            self.assertTrue(len(files) >= 1)

            # EMA consolidation from logged records.
            records = []
            for p in files:
                records.extend(torch.load(p))
            if records:
                model.run_cms_consolidation_ema(records, ema=0.9)

            # CMS shard export/import roundtrip.
            shard_dir = os.path.join(td, "shards")
            keys = model.save_cms_shards(shard_dir, shard_size=16, quantize=True)
            self.assertTrue(len(keys) > 0)
            self.assertTrue(os.path.exists(os.path.join(shard_dir, "manifest.pt")))
            model.load_cms_shards(shard_dir)

            # Checkpoint save/load roundtrip.
            ckpt_path = os.path.join(td, "broad_smoke_ckpt.pt")
            model.save_checkpoint(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))
            model.load_checkpoint(ckpt_path, strict=False)

            # Metrics/holo diagnostic smoke.
            metrics = model.get_metrics()
            self.assertIn("energy_mode", metrics)
            holo = model.get_holonomy_stats(x)
            self.assertIn("hg", holo)
            self.assertIn("cgmn", holo)

    def test_benchmark_wrapper_forward_smoke(self):
        vocab, bsz, seq, d_model = 80, 2, 5, 24
        model = CortexSeqModel(vocab_size=vocab, d_model=d_model, cms_enabled=True, cms_senses=3)
        src = torch.randint(0, vocab, (bsz, seq))

        model.train()
        logits, aux = model(src, return_aux_losses=True)
        self.assertEqual(tuple(logits.shape), (bsz, seq, vocab))
        self.assertIn("recall_loss", aux)
        self.assertIn("cms_loss", aux)

        model.eval()
        with torch.no_grad():
            logits_eval = model(src, return_aux_losses=False)
        self.assertEqual(tuple(logits_eval.shape), (bsz, seq, vocab))


if __name__ == "__main__":
    unittest.main()
