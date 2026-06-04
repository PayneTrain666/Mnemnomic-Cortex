import json
import torch

from mnemonic_cortex.reasoning_depth import (
    MultiPassThoughtPlanner,
    MultiPassThoughtPlannerConfig,
    EvidenceGuidedRouteExpanderConfig,
    PlannerEvaluator,
    PlannerEvaluationConfig,
)


def test_reason3b_planner_evaluation_enabled_scores_report():
    planner = MultiPassThoughtPlanner(
        MultiPassThoughtPlannerConfig(enabled=True, route_expander_config=EvidenceGuidedRouteExpanderConfig.enabled_default())
    )
    plan = planner.plan(torch.randn(1, 2, 8), content="evaluate")
    report = PlannerEvaluator(PlannerEvaluationConfig.enabled_default()).evaluate(plan, lineage={"stage": "test"})
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["score"]["pass_count"] >= 1
    assert payload["score"]["boundedness_ok"] is True
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
