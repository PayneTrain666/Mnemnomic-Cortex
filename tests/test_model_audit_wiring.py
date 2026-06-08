import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.model_audit import run_model_audit
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


def test_run_model_audit_static_inventory_without_probe():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    report = run_model_audit(model, sample_batch=None, auto_probe=False)
    assert report["probe"]["used_sample_batch"] is False
    assert report["utilization"]["total_modules"] >= 3
    assert "Model Audit Report" in report["text_report"]
    assert report["recommendations"]


def test_cortex_seq_model_run_full_audit_auto_probe():
    torch.manual_seed(0)
    model = CortexSeqModel(vocab_size=32, d_model=32, cms_enabled=False)
    report = model.run_full_audit(auto_probe=True, probe_batch_size=1, probe_seq_len=4)
    assert report["model_type"] == "CortexSeqModel"
    assert report["probe"]["used_sample_batch"] is True
    assert isinstance(report["layers"], list) and report["layers"]
    assert "Recommendations:" in report["text_report"]


def test_triple_hybrid_audit_hidden_layer_utilization():
    torch.manual_seed(0)
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        hg_slots=16,
        cgmn_slots=16,
        curved_slots=8,
    )
    report = model.ensure_hidden_layer_utilization()
    util = report["utilization"]
    assert util["total_modules"] > 0
    assert util["active_modules"] > 0
    assert 0.0 <= util["utilization_ratio"] <= 1.0
