import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from mnemonic_cortex.working_memory.qspin_prod7_observability import *

def test_observability_rejects_unsafe_span():
    c=build_default_prod7_observability_collector()
    with pytest.raises(ValueError):
        c.emit_span(Prod7SpanEvent('s','bad',contains_secret=True))

def test_duplicate_audit_idempotent():
    c=build_default_prod7_observability_collector()
    c.emit_audit(Prod7AuditEvent('a','x'))
    c.emit_audit(Prod7AuditEvent('a','x'))
    assert len(c.snapshot().audits) == 1
