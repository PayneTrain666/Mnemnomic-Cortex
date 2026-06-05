import json

from mnemonic_cortex.reasoning_depth import PlannerEvaluator, PlannerEvaluationConfig


def test_reason3b_planner_evaluation_disabled_is_inert():
    report = PlannerEvaluator(PlannerEvaluationConfig.disabled()).evaluate(None)
    payload = report.to_dict()

    assert payload["enabled"] is False
    assert payload["failure_signals"] == ["evaluation_disabled"]
    assert payload["safety_flags"]["automatic_remediation"] is False
    json.dumps(payload)
