import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_synthetic_real_boundary import *

def test_boundary_verifier_blocks_crossing():
    v=SyntheticToRealBoundaryVerifier()
    rec=BoundarySurfaceRecord('live', BoundarySurfaceKind.LIVE_ROUTE, real_side_touched=False)
    res=v.verify(BoundaryVerificationRequest('r', rec, live_runtime_requested=True))
    assert res.status == BoundaryVerificationStatus.BLOCKED
    assert BoundaryBlockReason.LIVE_RUNTIME_REQUESTED in res.reasons

def test_default_boundaries_pass_as_not_crossed():
    s=SyntheticToRealBoundaryVerifier().verify_suite(build_default_boundary_surface_records())
    assert s.failed == 0
    assert s.passed == len(BoundarySurfaceKind)
