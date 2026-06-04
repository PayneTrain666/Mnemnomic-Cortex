import json
import torch

from mnemonic_cortex.reasoning_depth import MultiPassThoughtPlanner, MultiPassThoughtPlannerConfig


def test_reason3a_planner_disabled_default_is_inert():
    planner = MultiPassThoughtPlanner(MultiPassThoughtPlannerConfig.disabled())
    x = torch.randn(1, 2, 8)
    before = x.clone()
    report = planner.plan(x, content="disabled")
    payload = report.to_dict()

    assert payload["enabled"] is False
    assert payload["passes"] == []
    assert torch.equal(x, before)
    json.dumps(payload)
