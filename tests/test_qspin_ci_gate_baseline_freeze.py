from qspin_prod8_module_loader import load_module
m = load_module("qspin_ci_gate_baseline_freeze")

def test_ci_gate_freeze_passes_defaults_and_blocks_missing():
    freezer = m.CIGateBaselineFreezer()
    ok = freezer.freeze(m.CIBaselineFreezeRequest("ok", m.build_default_ci_baseline_gate_records()))
    assert ok.status.value == "frozen"
    bad = freezer.freeze(m.CIBaselineFreezeRequest("bad", m.build_default_ci_baseline_gate_records()[:-1]))
    assert bad.status.value == "blocked"
