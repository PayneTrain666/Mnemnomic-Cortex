import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_runtime_observability import ObservabilityEmitter, no_secret_leakage_check

def test_redaction_and_summary():
    emitter = ObservabilityEmitter()
    emitter.emit_audit("x", "PASS", "OK", detail={"api_key": "SECRET"})
    summary = emitter.summary()
    assert no_secret_leakage_check(summary)
    assert summary["counters"]["audit_events"] == 1
