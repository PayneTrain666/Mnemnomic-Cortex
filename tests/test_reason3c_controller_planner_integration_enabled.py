import json
import torch

from mnemonic_cortex.reasoning_depth import ControllerPlannerIntegration, ControllerPlannerIntegrationConfig


def test_reason3c_controller_planner_integration_enabled():
    integration = ControllerPlannerIntegration(ControllerPlannerIntegrationConfig.enabled_default())
    x = torch.randn(1, 2, 8)
    before = x.clone()
    report = integration.run(x, content="integrate")
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["planner_report"]["enabled"] is True
    assert payload["evaluation_report"]["enabled"] is True
    assert payload["hardening_report"]["enabled"] is True
    assert torch.equal(x, before)
    json.dumps(payload)
