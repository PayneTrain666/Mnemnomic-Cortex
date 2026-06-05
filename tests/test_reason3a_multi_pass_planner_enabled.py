import json
import torch

from mnemonic_cortex.reasoning_depth import (
    EvidenceReasoningPass,
    EvidenceReasoningConfig,
    MultiPassThoughtPlanner,
    MultiPassThoughtPlannerConfig,
    EvidenceGuidedRouteExpanderConfig,
)


def test_reason3a_planner_enabled_produces_bounded_passes():
    evidence = EvidenceReasoningPass(EvidenceReasoningConfig.enabled_default()).run(content="Evidence one. Evidence two.")
    cfg = MultiPassThoughtPlannerConfig(
        enabled=True,
        max_passes=2,
        max_routes_per_pass=2,
        route_expander_config=EvidenceGuidedRouteExpanderConfig.enabled_default(),
    )
    planner = MultiPassThoughtPlanner(cfg)
    report = planner.plan(torch.randn(1, 2, 8), content="Plan this", evidence_report=evidence)
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert len(payload["passes"]) <= 2
    assert payload["safety"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
