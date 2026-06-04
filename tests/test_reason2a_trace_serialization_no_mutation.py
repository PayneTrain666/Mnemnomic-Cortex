import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2a_trace_serializes_and_input_not_mutated():
    controller = ReasoningController(ReasoningControllerConfig.enabled_default(key_dim=16, value_dim=16, slot_count=8))
    x = torch.randn(2, 3, 16)
    before = x.clone()
    result = controller.run_reasoning_pass(x, content="no mutation")
    assert torch.equal(x, before)
    payload = result.to_dict()
    assert payload["mutation_performed"] is False
    assert payload["safety"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
