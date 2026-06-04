import json
import torch

from mnemonic_cortex.reasoning_depth import MultiPassThoughtPlanner, MultiPassThoughtPlannerConfig


def test_reason3a_no_mutation_and_trace_serialization():
    planner = MultiPassThoughtPlanner(MultiPassThoughtPlannerConfig.enabled_default())
    x = torch.randn(1, 2, 8)
    before = x.clone()
    report = planner.plan(x, content="trace")
    payload = report.to_dict()

    assert torch.equal(x, before)
    assert payload["paamax_metadata"]["trace_governance"] is True
    assert payload["safety"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
