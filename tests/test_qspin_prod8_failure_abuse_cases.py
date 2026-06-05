from qspin_prod8_module_loader import load_module
import pytest
ready = load_module("qspin_final_readiness_review")
obs = load_module("qspin_prod8_observability")
ci = load_module("qspin_ci_gate_baseline_freeze")

def test_abuse_cases_block_activation_commit_and_secrets():
    reviewer = ready.FinalPreActivationReadinessReviewer()
    result = reviewer.review(ready.FinalReadinessReviewRequest("r", ready.build_default_final_readiness_evidence(), 0, production_activation_requested=True, commit_requested=True))
    reasons = {r.value for r in result.block_reasons}
    assert "production_activation_requested" in reasons
    assert "commit_requested" in reasons
    with pytest.raises(ValueError):
        obs.Prod8DeadLetterEvent("d", ("x",), secret_free=False).validate()
    bad_gate = ci.CIBaselineGateRecord(ci.CIBaselineGateKind.NO_WRITE, False, True, "write attempted")
    records = tuple([bad_gate] + [g for g in ci.build_default_ci_baseline_gate_records() if g.kind != ci.CIBaselineGateKind.NO_WRITE])
    freeze = ci.CIGateBaselineFreezer().freeze(ci.CIBaselineFreezeRequest("bad", records))
    assert freeze.status.value == "blocked"
