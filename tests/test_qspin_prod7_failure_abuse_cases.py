import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_readonly_runtime_probe import *
from mnemonic_cortex.working_memory.qspin_ci_gate_enforcement import *

def test_abuse_live_route_and_payload_transfer_rejected():
    h=ReadOnlyRuntimeProbeHarness(root='.')
    r=h.probe(ReadOnlyProbeRequest('abuse', ReadOnlyProbeTargetKind.NO_LIVE_ROUTE_SENTINEL, 'x', live_route_requested=True, payload_transfer_requested=True))
    assert r.status == ReadOnlyProbeStatus.BLOCKED
    assert ReadOnlyProbeBlockReason.LIVE_ROUTE_REQUESTED in r.reasons
    assert ReadOnlyProbeBlockReason.PAYLOAD_TRANSFER_REQUESTED in r.reasons

def test_ci_gate_fails_safety_boundary():
    r=CIGateEnforcer().run([CIGateCheck('prod_activation','prod activation', CIGateSeverity.CRITICAL, False)])
    assert r.failed == 1
