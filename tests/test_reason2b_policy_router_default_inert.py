import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningPolicyRouter, ReasoningPolicyRouterConfig


def test_reason2b_policy_router_default_inert_metadata():
    router = ReasoningPolicyRouter(ReasoningPolicyRouterConfig.disabled())
    decision = router.route(torch.randn(2, 3, 16), content="project episode")
    payload = decision.to_dict()

    assert payload["enabled"] is False
    assert payload["safety"]["non_mutating"] is True
    assert payload["route_plan"]["selected_depths"]
    assert payload["paamax_metadata"]["policy_lane_integration"] is True
    json.dumps(payload)
