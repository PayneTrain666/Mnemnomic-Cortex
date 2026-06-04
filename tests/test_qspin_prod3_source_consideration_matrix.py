from pathlib import Path

def test_prod3_source_consideration_matrix_mentions_all_required_families():
    text = Path("docs/qdt_wm_maae/qspin_prod3_qd6a_source_consideration_matrix.md").read_text(encoding="utf-8")
    for term in [
        "foundation/context", "depth/transformer", "attention", "external memory", "shared-slot/QH",
        "commit/cortex/compatibility", "guards", "quality subsystem", "QSPIN-0 through QSPIN-8",
        "PROD-0", "PROD-1", "PROD-2", "PROD-3", "context_triplet_projector.py",
        "wm_quantum_holographic_storage.py", "wm_system_commit_gate.py", "production caveats",
    ]:
        assert term in text
