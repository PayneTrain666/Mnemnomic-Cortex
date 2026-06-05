import json

from mnemonic_cortex.reasoning_depth import ReasoningReleaseCandidate, ReasoningReleaseCandidateConfig


def test_reason3d_release_candidate_report_json_safe():
    report = ReasoningReleaseCandidate(ReasoningReleaseCandidateConfig.enabled_default()).evaluate(lineage={"stage": "test"})
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["safety_flags"]["no_fake_production_complete_claim"] is True
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
    assert "production persistence adapter implementation" in payload["deferred_items"]
    json.dumps(payload)
