import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2c_disabled_default_remains_passthrough():
    controller = ReasoningController(ReasoningControllerConfig.disabled(key_dim=16))
    x = torch.randn(2, 3, 16)
    result = controller.run_reasoning_pass(x, content="disabled")
    payload = result.to_dict()
    stages = [event["stage"] for event in payload["trace"]["events"]]

    assert result.output is x
    assert "evidence_reasoning_pass" not in stages
    assert "counterfactual_reasoning_probe" not in stages
    assert payload["mutation_performed"] is False
    json.dumps(payload)
