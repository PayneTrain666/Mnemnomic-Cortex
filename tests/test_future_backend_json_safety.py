import json

from mnemonic_cortex.reasoning_depth import (
    BackendAuthorizationGate,
    BackendAuthorizationConfig,
    BackendThreatModelBuilder,
    BackendThreatModelConfig,
    BackendImplementationPlanner,
    BackendImplementationPlanConfig,
)


def test_future_backend_all_reports_json_safe():
    payloads = [
        BackendAuthorizationGate(BackendAuthorizationConfig.planning_authorized()).decide().to_dict(),
        BackendThreatModelBuilder(BackendThreatModelConfig.enabled_default()).build().to_dict(),
        BackendImplementationPlanner(BackendImplementationPlanConfig.enabled_default()).build().to_dict(),
    ]
    for payload in payloads:
        json.dumps(payload, sort_keys=True)
