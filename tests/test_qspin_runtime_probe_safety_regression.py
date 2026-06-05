import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_runtime_probe_safety_regression import *

def test_probe_safety_regression_default_cases_pass():
    result=ProbeSafetyRegressionRunner().run(build_default_probe_safety_cases())
    assert result.failed == 0
    assert result.passed == len(ProbeSafetyCaseKind)
