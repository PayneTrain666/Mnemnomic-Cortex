from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
obs = mods["qspin_prod3_observability"]

def test_prod3_observability_records_safe_metadata():
    c = obs.build_default_qspin_prod3_observability_collector()
    c.emit_metric(obs.QSpinProd3MetricRecord(obs.QSpinProd3MetricName.ACTIVE_DRY_RUN_EXECUTION_ATTEMPTS))
    c.emit_audit(obs.QSpinProd3AuditEvent("a", "act", ("ok",)))
    c.emit_trace(obs.QSpinProd3TraceRecord("t", {"safe": True}))
    c.dead_letter(obs.QSpinProd3DeadLetterRecord("d", ("blocked",)))
    snap = c.snapshot()
    assert len(snap.metrics) == 1
    assert len(snap.audit_events) == 1
    assert len(snap.traces) == 1
    assert len(snap.dead_letters) == 1

def test_prod3_observability_idempotent_duplicate_ids_and_rejects_unsafe():
    c = obs.build_default_qspin_prod3_observability_collector()
    c.emit_audit(obs.QSpinProd3AuditEvent("a", "act"))
    c.emit_audit(obs.QSpinProd3AuditEvent("a", "act2"))
    assert len(c.snapshot().audit_events) == 1
    try:
        c.emit_trace(obs.QSpinProd3TraceRecord("bad", {}, contains_secret=True))
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe trace should fail")
