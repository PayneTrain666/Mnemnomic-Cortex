import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_ci_gate_enforcement import *

def test_ci_gate_fails_critical_missing():
    r=CIGateEnforcer().run([CIGateCheck('no_live', 'no live', CIGateSeverity.CRITICAL, False)])
    assert r.failed == 1

def test_optional_pytest_skip():
    r=CIGateEnforcer().run([CIGateCheck('pytest', 'pytest', CIGateSeverity.OPTIONAL, False, True)])
    assert r.skipped == 1
