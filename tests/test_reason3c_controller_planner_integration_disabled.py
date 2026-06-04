import json
import torch

from mnemonic_cortex.reasoning_depth import ControllerPlannerIntegration, ControllerPlannerIntegrationConfig


def test_reason3c_controller_planner_integration_disabled():
    integration = ControllerPlannerIntegration(ControllerPlannerIntegrationConfig.disabled())
    x = torch.randn(1, 2, 8)
    before = x.clone()
    report = integration.run(x, content="disabled")
    payload = report.to_dict()

    assert payload["enabled"] is False
    assert torch.equal(x, before)
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
