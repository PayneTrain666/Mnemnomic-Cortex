import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_observability_review import *

def test_observability_rejects_secret():
    rec=ObservabilitySignalRecord('s', ObservabilitySignalKind.SECRET_REDACTION, True, contains_secret=True)
    res=ObservabilityReviewEngine().review(ObservabilityReviewRequest('r', (rec,)))
    assert res.status == ObservabilityReviewStatus.BLOCKED

def test_default_observability_passes():
    res=ObservabilityReviewEngine().review(ObservabilityReviewRequest('r', build_default_observability_signal_records()))
    assert res.status == ObservabilityReviewStatus.PASSED
