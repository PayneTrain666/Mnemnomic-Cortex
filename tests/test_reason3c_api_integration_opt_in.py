import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason3c_api_integration_opt_in():
    cfg = ReasoningControllerAPIConfig(
        enabled=True,
        key_dim=8,
        value_dim=8,
        slot_count=8,
        allow_controller_planner_integration=True,
    )
    api = ReasoningControllerAPI(cfg)
    result = api.run_reasoning_pass(torch.randn(1, 2, 8), content="api")
    payload = result.to_dict()

    stages = [event["stage"] for event in payload["reasoning_result"]["trace"]["events"]]
    assert "controller_planner_integration" in stages
    assert payload["api_config"]["allow_controller_planner_integration"] is True
    json.dumps(payload)
