import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_safety_regression_ci import run_safety_regressions, safety_summary_to_junit_xml

def test_safety_regressions_pass_and_xml_generates():
    summary = run_safety_regressions()
    assert summary["fail_count"] == 0
    assert "testsuite" in safety_summary_to_junit_xml(summary)
