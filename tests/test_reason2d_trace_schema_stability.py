import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason2d_trace_schema_stability_json_safe():
    api = ReasoningControllerAPI(ReasoningControllerAPIConfig(enabled=True, key_dim=8, value_dim=8, slot_count=4))
    result = api.run_reasoning_pass(torch.randn(1, 2, 8), content="trace")
    payload = result.to_dict()
    trace = payload["reasoning_result"]["trace"]

    assert "events" in trace
    assert "metadata" in trace
    json.dumps(trace, sort_keys=True)
