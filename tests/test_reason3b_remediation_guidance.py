import json

from mnemonic_cortex.reasoning_depth import (
    PlannerFailureClassifier,
    PlannerFailureClassifierConfig,
    PlannerEvaluator,
    PlannerEvaluationConfig,
    PlannerRemediationGuidance,
    PlannerRemediationGuidanceConfig,
)


def test_reason3b_remediation_guidance_recommendation_only():
    evaluation = PlannerEvaluator(PlannerEvaluationConfig.enabled_default()).evaluate({"enabled": True, "report_id": "p", "passes": [], "safety": {}})
    records = PlannerFailureClassifier(PlannerFailureClassifierConfig.enabled_default()).classify(evaluation)
    guidance = PlannerRemediationGuidance(PlannerRemediationGuidanceConfig.enabled_default()).recommend(records)
    payload = guidance.to_dict()

    assert payload["enabled"] is True
    assert payload["items"]
    assert payload["safety_flags"]["recommendation_only"] is True
    assert payload["safety_flags"]["allow_patch_application"] is False
    json.dumps(payload)
