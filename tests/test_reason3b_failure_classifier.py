from mnemonic_cortex.reasoning_depth import (
    PlannerEvaluator,
    PlannerEvaluationConfig,
    PlannerFailureClassifier,
    PlannerFailureClassifierConfig,
)


def test_reason3b_failure_classifier_detects_empty_plan_low_confidence():
    evaluation = PlannerEvaluator(PlannerEvaluationConfig.enabled_default()).evaluate({"enabled": True, "report_id": "p", "passes": [], "safety": {}})
    records = PlannerFailureClassifier(PlannerFailureClassifierConfig.enabled_default()).classify(evaluation)

    families = {record.family.value for record in records}
    assert "empty_plan" in families
    assert "low_confidence" in families
    assert all(record.failure_id.startswith("planner_failure_") for record in records)
