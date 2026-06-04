import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_synthetic_canary_bridge import run_canaries

def test_all_default_canaries_pass():
    results = run_canaries(seed=5005)
    assert len(results) >= 12
    assert all(r.status == "PASS" for r in results)
