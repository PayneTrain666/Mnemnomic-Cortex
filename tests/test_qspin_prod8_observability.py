from qspin_prod8_module_loader import load_module
import pytest
m = load_module("qspin_prod8_observability")

def test_prod8_observability_rejects_unsafe_and_idempotent_audit():
    c = m.build_default_prod8_observability_collector()
    c.emit_audit(m.Prod8AuditEvent("a", "x"))
    c.emit_audit(m.Prod8AuditEvent("a", "x"))
    assert len(c.snapshot().audits) == 1
    with pytest.raises(ValueError):
        c.emit_audit(m.Prod8AuditEvent("bad", "x", raw_payload_free=False))
