import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_readonly_runtime_probe import *

def test_readonly_probe_blocks_writes():
    h=ReadOnlyRuntimeProbeHarness(root='.')
    r=h.probe(ReadOnlyProbeRequest('x', ReadOnlyProbeTargetKind.NO_WRITE_SENTINEL, 'no_write', write_requested=True))
    assert r.status == ReadOnlyProbeStatus.BLOCKED
    assert ReadOnlyProbeBlockReason.WRITE_REQUESTED in r.reasons

def test_default_probe_suite_runs():
    s=ReadOnlyRuntimeProbeHarness(root='.').run_suite(build_default_readonly_probe_suite())
    assert s.failed == 0
    assert s.passed >= 5
