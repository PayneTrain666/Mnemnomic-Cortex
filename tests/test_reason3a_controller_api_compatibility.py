import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason3a_controller_api_planner_opt_in_and_backward_compatible():
    disabled = ReasoningControllerAPI(ReasoningControllerAPIConfig.disabled(key_dim=8))
    disabled_payload = disabled.run_reasoning_pass(torch.randn(1, 2, 8), content="x").to_dict()
    assert disabled_payload["api_config"]["enabled"] is False

    cfg = ReasoningControllerAPIConfig(
        enabled=True,
        key_dim=8,
        value_dim=8,
        slot_count=8,
        allow_multi_pass_planner=True,
    )
    api = ReasoningControllerAPI(cfg)
    payload = api.run_reasoning_pass(torch.randn(1, 2, 8), content="planner").to_dict()
    stages = [event["stage"] for event in payload["reasoning_result"]["trace"]["events"]]
    assert "multi_pass_thought_planner" in stages
    json.dumps(payload)
