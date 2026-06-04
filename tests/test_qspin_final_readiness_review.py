from qspin_prod8_module_loader import load_module
m = load_module("qspin_final_readiness_review")

def test_final_readiness_blocks_production_ready_with_blockers():
    reviewer = m.FinalPreActivationReadinessReviewer()
    result = reviewer.review(m.FinalReadinessReviewRequest("r", m.build_default_final_readiness_evidence(), critical_blockers_open=1))
    assert result.production_ready is False
    assert result.hold_required is True
    assert any(x.value == "critical_blockers_open" for x in result.block_reasons)
