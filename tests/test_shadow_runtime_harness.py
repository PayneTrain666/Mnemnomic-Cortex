import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_shadow_runtime_harness import ShadowRuntimeEnvelope, ShadowRuntimeHarness

def test_valid_shadow_passes():
    result = ShadowRuntimeHarness().execute(ShadowRuntimeEnvelope(payload={"ok": True}))
    assert result.status == "PASS"
    assert result.diagnostics["persistent_writes"] is False

def test_qh_write_blocks_fail_closed():
    result = ShadowRuntimeHarness().execute(ShadowRuntimeEnvelope(payload={"x": 1}, flags={"write_qh": True}))
    assert result.status == "FAIL_CLOSED"
    assert "forbidden_flag:write_qh" in result.blocked_actions
