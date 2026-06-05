import os
import tempfile
import unittest

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.train_smoke import smoke_run, tiny_train_step
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


class TestExtendedSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)

    def test_no_cms_fallback_path(self):
        bsz, seq, dim = 2, 4, 24
        model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=0)
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)

        # Should run fine without token_ids and with CMS disabled.
        y_proc = model(x, ctx, operation="process")
        y_ret = model(x, ctx, operation="retrieve")
        self.assertEqual(tuple(y_proc.shape), (bsz, seq, dim))
        self.assertEqual(tuple(y_ret.shape), (bsz, dim))
        self.assertIsNone(model.consolidated_lexicon)

    def test_triple_hybrid_write_read_smoke(self):
        bsz, seq, dim = 2, 3, 16
        mem = EnhancedTripleHybridMemory(
            input_dim=dim,
            output_dim=dim,
            hg_slots=48,
            cgmn_slots=32,
            curved_slots=24,
            fusion="weighted",
        )
        x = torch.randn(bsz, seq, dim)
        _ = mem(x, operation="write")
        y = mem(x, operation="read")
        self.assertEqual(tuple(y.shape), (bsz, seq, dim))

    def test_cms_shard_non_quantized_roundtrip(self):
        bsz, seq, dim, vocab = 2, 3, 16, 48
        model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=vocab, cms_senses=2)
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)
        token_ids = torch.randint(0, vocab, (bsz, seq))
        _ = model(
            x,
            ctx,
            operation="process",
            token_ids=token_ids,
            use_consolidated_memory=True,
            context_features=x,
        )

        with tempfile.TemporaryDirectory() as td:
            keys = model.save_cms_shards(td, shard_size=12, quantize=False)
            self.assertTrue(len(keys) > 0)
            self.assertTrue(os.path.exists(os.path.join(td, "manifest.pt")))
            model.load_cms_shards(td, keys=keys[:1])  # subset load path smoke

    def test_train_smoke_entrypoints(self):
        proc_shape, ret_shape = smoke_run(device="cpu")
        self.assertEqual(proc_shape, torch.Size([8, 5, 128]))
        self.assertEqual(ret_shape, torch.Size([8, 128]))
        final_loss = tiny_train_step(steps=1, device="cpu")
        self.assertTrue(isinstance(final_loss, float))
        self.assertTrue(torch.isfinite(torch.tensor(final_loss)))


if __name__ == "__main__":
    unittest.main()
