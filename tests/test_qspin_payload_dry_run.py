from tests.qspin_prod2_module_loader import load_prod2_modules
m=load_prod2_modules()["qspin_payload_dry_run"]
def good_req():
    env=m.QSpinPayloadDryRunEnvelopeSummary("dense","src","dst",m.QSpinPayloadDryRunShapeSummary((2,3)),m.QSpinPayloadDryRunBudgetSummary(24),(0.0,2.0))
    return m.QSpinPayloadDryRunRequest("p", env)
def test_safe_summary_and_hash_deterministic():
    r=m.QSpinTraceSafePayloadSummarizer().dry_run(good_req()); r2=m.QSpinTraceSafePayloadSummarizer().dry_run(good_req())
    assert r.decision.approved and r.trace.safe_summary["safe_hash"]==r2.trace.safe_summary["safe_hash"] and not r.transferred_payload
def test_raw_payload_and_tensor_rejected():
    env=good_req().envelope
    assert not m.QSpinTraceSafePayloadSummarizer().dry_run(m.QSpinPayloadDryRunRequest("bad", env, raw_payload={"x":1})).decision.approved
    assert not m.QSpinTraceSafePayloadSummarizer().dry_run(m.QSpinPayloadDryRunRequest("bad2", env, raw_tensor=object())).decision.approved
def test_unsafe_shape_and_budget_rejected():
    env=m.QSpinPayloadDryRunEnvelopeSummary("dense","src","dst",m.QSpinPayloadDryRunShapeSummary((0,)),m.QSpinPayloadDryRunBudgetSummary(1))
    assert not m.QSpinTraceSafePayloadSummarizer().dry_run(m.QSpinPayloadDryRunRequest("shape", env)).decision.approved
    env2=m.QSpinPayloadDryRunEnvelopeSummary("dense","src","dst",m.QSpinPayloadDryRunShapeSummary((2,)),m.QSpinPayloadDryRunBudgetSummary(999, max_bytes=2))
    assert not m.QSpinTraceSafePayloadSummarizer().dry_run(m.QSpinPayloadDryRunRequest("budget", env2)).decision.approved
