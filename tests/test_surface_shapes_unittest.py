import tempfile
import unittest

import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


class TestSurfaceShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_cortex_process_retrieve_shapes_with_cms(self):
        bsz, seq, dim = 3, 5, 32
        model = EnhancedMnemonicCortex(
            input_dim=dim,
            output_dim=dim,
            cms_vocab_size=128,
            cms_senses=3,
        )
        x = torch.randn(bsz, seq, dim)
        ctx = torch.randn(bsz, dim)
        ids = torch.randint(0, 128, (bsz, seq))

        y_proc = model(
            x,
            ctx,
            operation="process",
            token_ids=ids,
            use_consolidated_memory=True,
            context_features=x,
        )
        y_ret = model(
            x,
            ctx,
            operation="retrieve",
            token_ids=ids,
            use_consolidated_memory=True,
            context_features=x,
        )

        self.assertEqual(tuple(y_proc.shape), (bsz, seq, dim))
        self.assertEqual(tuple(y_ret.shape), (bsz, dim))
        self.assertIsNotNone(model.last_cms_aux)
        self.assertIn("weights", model.last_cms_aux)
        self.assertIn("cue_h", model.last_cms_aux)
        self.assertEqual(model.last_cms_aux["weights"].shape[-1], 3)

    def test_benchmark_cortexseq_aux_shapes(self):
        vocab, bsz, seq, d_model = 96, 2, 6, 32
        model = CortexSeqModel(vocab_size=vocab, d_model=d_model, cms_enabled=True, cms_senses=3)
        model.train()
        src = torch.randint(0, vocab, (bsz, seq))
        logits, aux = model(src, return_aux_losses=True)
        self.assertEqual(tuple(logits.shape), (bsz, seq, vocab))
        self.assertIn("recall_loss", aux)
        self.assertIn("cms_loss", aux)

    def test_cms_logger_flush_creates_file(self):
        vocab, bsz, seq, d_model = 64, 2, 4, 16
        with tempfile.TemporaryDirectory() as td:
            model = CortexSeqModel(
                vocab_size=vocab,
                d_model=d_model,
                cms_enabled=True,
                cms_log_dir=td,
                cms_log_sample_rate=1.0,
                cms_log_max_buffer=1,
            )
            src = torch.randint(0, vocab, (bsz, seq))
            _ = model(src, return_aux_losses=False)
            flushed = model.flush_cms_logger()
            self.assertTrue(flushed is None or flushed.endswith(".pt"))

    def test_topology_policy_wires_qhm_knobs(self):
        dim = 24
        model = EnhancedMnemonicCortex(input_dim=dim, output_dim=dim, cms_vocab_size=64, cms_senses=2)
        model.topology.register_policy(
            "surface_check",
            qhm_enable=False,
            qhm_temp=0.77,
            qhm_phase_noise=0.05,
            qhm_lightbulb=0.91,
            qhm_explosive_temp=0.52,
            qhm_explosive_alpha=0.61,
        )
        model.apply_topology_policy("surface_check")
        self.assertFalse(model.working_memory.qhm_enabled)
        self.assertAlmostEqual(float(model.working_memory.holo.temperature.item()), 0.77, places=5)


if __name__ == "__main__":
    unittest.main()
