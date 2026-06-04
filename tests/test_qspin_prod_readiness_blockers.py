import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_prod_readiness_blockers import *

def test_readiness_blocks_production_active_claim():
    report=build_default_readiness_blocker_register().report()
    assert report.open_critical > 0
    assert report.production_active_allowed is False
