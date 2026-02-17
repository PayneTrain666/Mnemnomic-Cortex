import unittest

import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.anti_hallucination import AHGThresholds


class TestBenchmarkGuardWiring(unittest.TestCase):
    def test_guarded_infer_wired(self):
        torch.manual_seed(5)
        vocab, bsz, seq, d_model = 64, 2, 4, 24
        model = CortexSeqModel(vocab_size=vocab, d_model=d_model, ahg_enabled=True)
        src = torch.randint(0, vocab, (bsz, seq))

        model.enable_hallucination_guard(
            thresholds=AHGThresholds(min_retrieval_score=0.0, min_coverage=0.0),
            retriever_fn=lambda q, k=5: [{"text": "doc", "score": 0.8, "source_id": "s1"}],
            cite_align_fn=lambda a, d: 0.8,
            cms_signals_fn=lambda t: {"sense_entropy": 0.2, "proto_distance": 0.1, "warp": 1.0},
        )
        out = model.guarded_infer(src, query_text="test question")
        self.assertIn(out["mode"], {"answer", "grounded", "ask", "refuse"})
        self.assertIn("logits", out)
        self.assertEqual(tuple(out["logits"].shape), (bsz, seq, vocab))


if __name__ == "__main__":
    unittest.main()
