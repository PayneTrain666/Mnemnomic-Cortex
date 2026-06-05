from mnemonic_cortex.working_memory.qspin_prod6_observability import build_default_prod6_observability_collector, Prod6AuditEvent, Prod6TraceRecord, Prod6MetricEvent, Prod6MetricName

def test_prod6_observability_rejects_unsafe_and_is_idempotent():
    c = build_default_prod6_observability_collector()
    c.emit_metric(Prod6MetricEvent(Prod6MetricName.STRESS_REPLAY_ATTEMPTS, 1))
    c.emit_audit(Prod6AuditEvent('a', 'act'))
    c.emit_audit(Prod6AuditEvent('a', 'act2'))
    assert len(c.snapshot().audits) == 1
    try:
        Prod6TraceRecord('t', {'raw_payload': 'bad'}).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe trace should fail')
