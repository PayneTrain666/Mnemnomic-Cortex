import glob
import os
import tempfile
import unittest

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex


class TestSurfaceSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_cms_shard_roundtrip_and_consolidation_ema(self):
        bsz, seq, dim, vocab = 2, 4, 16, 64
        model = EnhancedMnemonicCortex(
            input_dim=dim,
            output_dim=dim,
            cms_vocab_size=vocab,
            cms_senses=3,
        )
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)
        ids = torch.randint(0, vocab, (bsz, seq))

        # Produce CMS aux once.
        _ = model(
            x,
            ctx,
            operation="process",
            token_ids=ids,
            use_consolidated_memory=True,
            context_features=x,
        )
        self.assertIsNotNone(model.last_cms_aux)

        # Run a tiny consolidation update from synthetic records.
        flat_ids = ids.reshape(-1)
        aux = model.last_cms_aux
        records = []
        for i in range(min(8, flat_ids.numel())):
            records.append(
                {
                    "token_id": int(flat_ids[i].item()),
                    "sense_w": aux["weights"][i].detach().cpu().tolist(),
                    "cue_h": aux["cue_h"][i].detach().cpu().tolist(),
                    "cue_p": aux["cue_p"][i].detach().cpu().tolist(),
                    "cue_e": aux["cue_e"][i].detach().cpu().tolist(),
                }
            )
        model.run_cms_consolidation_ema(records, ema=0.9)

        # Shard save/load smoke.
        with tempfile.TemporaryDirectory() as td:
            keys = model.save_cms_shards(td, shard_size=16, quantize=True)
            self.assertTrue(len(keys) > 0)
            manifest = os.path.join(td, "manifest.pt")
            self.assertTrue(os.path.exists(manifest))
            model.load_cms_shards(td)

    def test_logger_pipeline_and_flush_files(self):
        bsz, seq, dim, vocab = 2, 3, 16, 32
        with tempfile.TemporaryDirectory() as td:
            model = EnhancedMnemonicCortex(
                input_dim=dim,
                output_dim=dim,
                cms_vocab_size=vocab,
                cms_senses=3,
            )
            model.enable_cms_logger(out_dir=td, max_buffer=2, sample_rate=1.0)
            x = torch.randn(bsz, seq, dim)
            ctx = torch.randn(bsz, dim)
            ids = torch.randint(0, vocab, (bsz, seq))

            _ = model(
                x,
                ctx,
                operation="process",
                token_ids=ids,
                use_consolidated_memory=True,
                context_features=x,
            )
            _ = model.flush_cms_logger()
            files = glob.glob(os.path.join(td, "cms_records_*.pt"))
            self.assertTrue(len(files) >= 1)


if __name__ == "__main__":
    unittest.main()
