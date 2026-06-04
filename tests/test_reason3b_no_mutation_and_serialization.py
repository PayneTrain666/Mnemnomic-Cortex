import json
import torch

from mnemonic_cortex.reasoning_depth import (
    MultiPassThoughtPlanner,
    MultiPassThoughtPlannerConfig,
    PlannerEvaluator,
    PlannerEvaluationConfig,
)


def test_reason3b_no_input_mutation_and_serialization():
    planner = MultiPassThoughtPlanner(MultiPassThoughtPlannerConfig.enabled_default())
    x = torch.randn(1, 2, 8)
    before = x.clone()
    plan = planner.plan(x, content="safe")
    report = PlannerEvaluator(PlannerEvaluationConfig.enabled_default()).evaluate(plan)

    assert torch.equal(x, before)
    json.dumps(report.to_dict())
