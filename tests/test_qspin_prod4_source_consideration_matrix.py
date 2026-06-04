from pathlib import Path

def test_source_consideration_matrix_mentions_required_families():
    text=Path("docs/qdt_wm_maae/qspin_prod4_qd6a_source_consideration_matrix.md").read_text(encoding="utf-8")
    for term in ["foundation/context","depth/transformer","attention","external memory","shared-slot/QH","commit/cortex/compatibility","guards","quality subsystem","QSPIN-0 through QSPIN-8","PROD-0","PROD-1","PROD-2","PROD-3","PROD-4","production caveats","runtime safety regression"]:
        assert term in text
