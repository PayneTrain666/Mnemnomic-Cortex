import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason2d_controller_api_disabled_passthrough_json_safe():
    api = ReasoningControllerAPI(ReasoningControllerAPIConfig.disabled(key_dim=8))
    x = torch.randn(1, 2, 8)
    result = api.run_reasoning_pass(x, content="disabled")
    payload = result.to_dict()

    assert payload["api_config"]["enabled"] is False
    assert payload["reasoning_result"]["output_shape"] == [1, 2, 8]
    assert payload["reasoning_result"]["mutation_performed"] is False
    json.dumps(payload)
