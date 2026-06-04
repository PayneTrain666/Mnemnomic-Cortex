import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_source_consideration_matrix import build_source_matrix, consideration_coverage_audit, matrix_to_markdown

def test_matrix_builds_with_coverage():
    records = build_source_matrix()
    audit = consideration_coverage_audit(records)
    assert "coverage_status" in audit
    assert "QD6A" in [r.source_id for r in records]
    assert "Source Consideration Matrix" in matrix_to_markdown(records)
