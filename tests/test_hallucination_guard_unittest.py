import unittest

import torch

from mnemonic_cortex.anti_hallucination import AHGThresholds, HallucinationGuard


class TestHallucinationGuard(unittest.TestCase):
    def test_allow_path(self):
        def gen(prompt, **kw):
            logits = torch.tensor([[[4.0, 1.0, -0.5], [3.5, 0.5, -0.2]]], dtype=torch.float32)
            return {"out_text": "Known answer.", "logits": logits, "tokens": torch.tensor([[1, 2]])}

        def retrieve(query, k=5):
            return [{"text": "evidence", "score": 0.9, "source_id": "doc-1"}]

        g = HallucinationGuard(
            thresholds=AHGThresholds(min_retrieval_score=0.0, min_coverage=0.0),
            generate_fn=gen,
            retriever_fn=retrieve,
            cite_align_fn=lambda a, d: 0.9,
            cms_signals_fn=lambda t: {"sense_entropy": 0.4, "proto_distance": 0.2, "warp": 1.0},
        )
        d = g.assess("q")
        self.assertEqual(d.action, "ALLOW")
        out = g.answer("q")
        self.assertEqual(out["mode"], "answer")

    def test_retrieve_or_ask_path(self):
        calls = {"n": 0}

        def gen(prompt, **kw):
            calls["n"] += 1
            # High entropy / low margin style logits.
            logits = torch.zeros(1, 2, 4, dtype=torch.float32)
            return {"out_text": "uncertain text", "logits": logits, "tokens": torch.tensor([[1, 1]])}

        g = HallucinationGuard(
            thresholds=AHGThresholds(),
            generate_fn=gen,
            retriever_fn=None,
            cite_align_fn=None,
            cms_signals_fn=lambda t: {"sense_entropy": 2.0, "proto_distance": 0.95, "warp": 1.0},
        )
        d = g.assess("q")
        self.assertIn(d.action, {"ASK", "REFUSE"})
        out = g.answer("q")
        self.assertIn(out["mode"], {"ask", "refuse"})


if __name__ == "__main__":
    unittest.main()
