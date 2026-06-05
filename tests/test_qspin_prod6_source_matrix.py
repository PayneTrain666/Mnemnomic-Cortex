from mnemonic_cortex.working_memory.qspin_prod6_source_matrix import build_prod6_source_matrix, Prod6SourceFamily

def test_prod6_source_matrix_complete():
    m = build_prod6_source_matrix()
    families = {r.family for r in m.records}
    assert set(Prod6SourceFamily) <= families
