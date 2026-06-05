import json

from mnemonic_cortex.reasoning_depth import BackendImplementationPlanner, BackendImplementationPlanConfig


def test_future_backend_implementation_plan_is_plan_only():
    report = BackendImplementationPlanner(BackendImplementationPlanConfig.enabled_default()).build().to_dict()

    assert report["enabled"] is True
    assert report["status"] == "ready_for_separate_authorization"
    assert report["write_capable_code_generated"] is False
    assert report["real_store_write_authorized"] is False
    assert "no_write_without_second_authorization" in report["acceptance_tests"]
    json.dumps(report)
