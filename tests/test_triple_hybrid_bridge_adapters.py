import torch
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


def test_triple_hybrid_ingest_episodic_vectors_writes_all_banks():
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        hg_slots=64,
        cgmn_slots=32,
        curved_slots=16,
    )
    x = torch.randn(6, 32)
    out = model.ingest_episodic_vectors(x, target_banks=("hg", "cgmn", "curved"))
    assert set(out.keys()) == {"hg", "cgmn", "curved"}
    assert out["hg"] > 0 and out["cgmn"] > 0 and out["curved"] > 0
    assert float(model.hg.usage_counts.sum().item()) > 0.0
    assert float(model.cgmn.usage_counts.sum().item()) > 0.0
    assert float(model.curved.usage_counts.sum().item()) > 0.0


def test_triple_hybrid_write_and_read_bank_helpers():
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        hg_slots=64,
        cgmn_slots=32,
        curved_slots=16,
    )
    x = torch.randn(2, 4, 32)
    model.write_bank("episodic", x)
    model.write_bank("semantic", x)
    model.write_bank("spatial", x)
    rhg = model.read_bank("hg", x)
    rcg = model.read_bank("cgmn", x)
    rcv = model.read_bank("curved", x)
    assert tuple(rhg.shape) == tuple(x.shape)
    assert tuple(rcg.shape) == tuple(x.shape)
    assert tuple(rcv.shape) == tuple(x.shape)
