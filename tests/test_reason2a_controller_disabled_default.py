import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2a_controller_disabled_default_passes_through():
    controller = ReasoningController(ReasoningControllerConfig.disabled(key_dim=16))
    x = torch.randn(2, 3, 16)
    result = controller.run_reasoning_pass(x, content="disabled", return_trace=True)

    assert result.output is x
    assert result.mutation_performed is False
    payload = result.to_dict()
    assert payload["mutation_performed"] is False
    assert payload["trace"]["events"][0]["message"] == "controller disabled; pass-through returned"
    json.dumps(payload)
