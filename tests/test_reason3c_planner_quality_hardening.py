import json

from mnemonic_cortex.reasoning_depth import (
    PlannerEvaluator,
    PlannerEvaluationConfig,
    PlannerFailureClassifier,
    PlannerFailureClassifierConfig,
    PlannerRemediationGuidance,
    PlannerRemediationGuidanceConfig,
    PlannerQualityHardener,
    PlannerQualityHardeningConfig,
)


def test_reason3c_planner_quality_hardening_report():
    evaluation = PlannerEvaluator(PlannerEvaluationConfig.enabled_default()).evaluate({"enabled": True, "report_id": "p", "passes": [], "safety": {}})
    failures = PlannerFailureClassifier(PlannerFailureClassifierConfig.enabled_default()).classify(evaluation)
    remediation = PlannerRemediationGuidance(PlannerRemediationGuidanceConfig.enabled_default()).recommend(failures)
    report = PlannerQualityHardener(PlannerQualityHardeningConfig.enabled_default()).harden(
        evaluation_report=evaluation,
        failure_records=failures,
        remediation_report=remediation,
    )
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["actions"]
    assert payload["safety_flags"]["automatic_patch_application"] is False
    assert payload["paamax_metadata"]["audit_metadata"] is True
    json.dumps(payload)
