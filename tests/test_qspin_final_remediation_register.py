from qspin_prod8_module_loader import load_module
m = load_module("qspin_final_remediation_register")

def test_final_remediation_register_preserves_hold():
    report = m.FinalRemediationReport.from_register(m.build_default_final_remediation_register())
    assert report.open_p0
    assert report.hold_required is True
