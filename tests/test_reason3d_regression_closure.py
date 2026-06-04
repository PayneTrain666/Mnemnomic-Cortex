import json

from mnemonic_cortex.reasoning_depth import ReasoningRegressionClosure, ReasoningRegressionClosureConfig


def test_reason3d_regression_closure_json_safe():
    report = ReasoningRegressionClosure(ReasoningRegressionClosureConfig.enabled_default()).close(lineage={"stage": "test"})
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert "REASON-3D" in payload["covered_stages"]
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
