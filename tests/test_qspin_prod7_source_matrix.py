import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_prod7_source_matrix import *

def test_source_matrix_covers_required_families():
    m=build_prod7_source_matrix()
    families={r.family for r in m.records}
    assert Prod7SourceFamily.QD6A in families
    assert Prod7SourceFamily.PROD7 in families
    assert 'PROD-7' in export_prod7_source_matrix_markdown(m)
