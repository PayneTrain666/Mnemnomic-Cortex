import json

from mnemonic_cortex.reasoning_depth import (
    EvidenceReasoningPass,
    EvidenceReasoningConfig,
    EvidenceGuidedRouteExpander,
    EvidenceGuidedRouteExpanderConfig,
)


def test_reason3a_evidence_guided_route_expander_scores_routes():
    evidence = EvidenceReasoningPass(EvidenceReasoningConfig.enabled_default()).run(content="A. B.")
    expander = EvidenceGuidedRouteExpander(EvidenceGuidedRouteExpanderConfig(enabled=True, max_route_candidates=2))
    routes = [
        {"route_id": "r1", "selected_depth_roles": ["Z4"], "confidence": 0.7, "disagreement": 0.1},
        {"route_id": "r2", "selected_depth_roles": ["Z6"], "confidence": 0.1, "disagreement": 0.9, "conflict_prone": True},
    ]
    report = expander.expand(routes, evidence_report=evidence)
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert len(payload["candidates"]) == 2
    assert any(item["conflict_prone"] for item in payload["candidates"])
    assert payload["paamax_metadata"]["confidence_hook"] is True
    json.dumps(payload)
