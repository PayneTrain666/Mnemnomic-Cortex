from qspin_prod8_module_loader import load_module
m = load_module("qspin_prod8_source_matrix")

def test_prod8_source_matrix_covers_required_families():
    matrix = m.build_prod8_source_matrix()
    families = {r.family for r in matrix.records}
    assert set(m.Prod8SourceFamily).issubset(families)
    assert "prod8" in m.export_prod8_source_matrix_json(matrix)
