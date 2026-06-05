from tests.qspin_prod4_module_loader import load_prod4_modules
m=load_prod4_modules()["qspin_prod4_observability"]

def test_observability_safe_and_idempotent():
    c=m.build_default_qspin_prod4_observability_collector()
    c.emit_metric(m.QSpinProd4MetricRecord(m.QSpinProd4MetricName.NO_MUTATION_ASSERTIONS_PASSED,1))
    c.emit_audit(m.QSpinProd4AuditEvent("a","act")); c.emit_audit(m.QSpinProd4AuditEvent("a","act"))
    c.emit_trace(m.QSpinProd4TraceRecord("t",{"safe":True}))
    c.dead_letter(m.QSpinProd4DeadLetterRecord("d",("reason",)))
    snap=c.snapshot()
    assert len(snap.audit_events)==1 and len(snap.metrics)==1

def test_unsafe_trace_rejected():
    try: m.QSpinProd4TraceRecord("bad",{},contains_secret=True).validate()
    except ValueError: pass
    else: raise AssertionError("unsafe trace should fail")
