from mnemonic_cortex.reasoning_depth import (
    PlannerEvaluator,
    PlannerEvaluationConfig,
    PlannerFailureClassifier,
    PlannerFailureClassifierConfig,
)


def test_reason3b_failure_classifier_caps_records():
    evaluation = PlannerEvaluator(PlannerEvaluationConfig(enabled=True, max_plan_passes=1, max_route_candidates=1)).evaluate(
        {
            "enabled": True,
            "report_id": "p",
            "passes": [{"pass_id": str(i), "confidence": 0.0, "disagreement": 1.0, "evidence_support": 0.0} for i in range(10)],
            "route_expansion": {"candidates": [{"route_id": str(i), "unsupported": True, "conflict_prone": True} for i in range(10)]},
            "safety": {},
        }
    )
    records = PlannerFailureClassifier(PlannerFailureClassifierConfig(enabled=True, max_failures=3, max_passes=1, max_routes=1)).classify(evaluation)

    assert len(records) <= 3
