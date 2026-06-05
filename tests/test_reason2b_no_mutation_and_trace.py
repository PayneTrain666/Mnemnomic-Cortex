import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig, ReasoningPolicyRouterConfig


def test_reason2b_policy_controller_does_not_mutate_input_and_trace_serializes():
    cfg = ReasoningControllerConfig(
        enabled=True,
        key_dim=16,
        value_dim=16,
        slot_count=8,
        use_policy_router=True,
        policy_router_config=ReasoningPolicyRouterConfig.enabled_default(),
    )
    controller = ReasoningController(cfg)
    x = torch.randn(2, 3, 16)
    before = x.clone()
    result = controller.run_reasoning_pass(x, content="trace no mutation")
    assert torch.equal(x, before)
    assert result.mutation_performed is False
    json.dumps(result.to_dict())
