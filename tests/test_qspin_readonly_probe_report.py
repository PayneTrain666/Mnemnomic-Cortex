from qspin_prod8_module_loader import load_module
m = load_module("qspin_readonly_probe_report")

def test_probe_report_contains_structured_findings():
    report = m.ReadOnlyRuntimeProbeReportGenerator().generate()
    assert report.sections
    assert "read_only" in report.to_markdown()
