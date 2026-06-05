from tests.qspin_prod2_module_loader import load_prod2_modules
m=load_prod2_modules()["qspin_prod2_observability"]
def test_observability_safe_and_idempotent():
    c=m.build_default_qspin_prod2_observability_collector(); c.emit_metric(m.QSpinProd2MetricRecord(m.QSpinProd2MetricName.SHADOW_BUS_DISPATCH_ATTEMPTS)); c.emit_audit(m.QSpinProd2AuditEvent("a","x")); c.emit_audit(m.QSpinProd2AuditEvent("a","x")); c.emit_trace(m.QSpinProd2TraceRecord("t",{"ok":True})); c.dead_letter(m.QSpinProd2DeadLetterRecord("d",("blocked",)))
    s=c.snapshot(); assert len(s.audit_events)==1 and len(s.metrics)==1
def test_unsafe_trace_deadletter_rejected():
    c=m.build_default_qspin_prod2_observability_collector()
    try: c.emit_trace(m.QSpinProd2TraceRecord("bad",{},contains_raw_payload=True))
    except ValueError: pass
    else: raise AssertionError("raw trace should fail")
    try: c.dead_letter(m.QSpinProd2DeadLetterRecord("bad",("x",),contains_secret=True))
    except ValueError: pass
    else: raise AssertionError("secret dead-letter should fail")
