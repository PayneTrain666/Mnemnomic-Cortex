import torch
import pytest

from mnemonic_cortex.reasoning_depth import ReasoningPolicyRouter, ReasoningPolicyRouterConfig, ReasoningPolicyRouterError


def test_reason2b_policy_router_rejects_non_finite_query():
    router = ReasoningPolicyRouter(ReasoningPolicyRouterConfig.enabled_default())
    bad = torch.zeros(1, 2, 8)
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ReasoningPolicyRouterError):
        router.route(bad)
