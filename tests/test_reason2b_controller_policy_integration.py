import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig, ReasoningPolicyRouterConfig


def test_reason2b_controller_policy_router_integration_event():
    cfg = ReasoningControllerConfig(
        enabled=True,
        key_dim=16,
        value_dim=16,
        slot_count=8,
        max_reasoning_hops=3,
        use_policy_router=True,
        policy_router_config=ReasoningPolicyRouterConfig.enabled_default(task_mode="hypothesis"),
    )
    controller = ReasoningController(cfg)
    result = controller.run_reasoning_pass(torch.randn(2, 3, 16), content="hypothesis project")

    payload = result.to_dict()
    stages = [event["stage"] for event in payload["trace"]["events"]]
    assert "reasoning_policy_router" in stages
    assert payload["mutation_performed"] is False
    assert payload["consolidation_evaluation"]["committed"] is False
    json.dumps(payload)
