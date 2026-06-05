import json

from mnemonic_cortex.reasoning_depth import run_reasoning_release_audit, ReasoningReleaseAuditConfig


def test_reason2d_release_audit_serializes_release_candidate():
    report = run_reasoning_release_audit(ReasoningReleaseAuditConfig(source_pack="unit-test"))
    payload = report.to_dict()

    assert payload["disabled_default_verified"] is True
    assert payload["no_mutation_verified"] is True
    assert payload["config_serialization_verified"] is True
    assert payload["trace_serialization_verified"] is True
    json.dumps(payload)
