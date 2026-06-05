import json

from mnemonic_cortex.reasoning_depth import ReasoningFinalSafetyAudit, FinalSafetyAuditConfig


def test_reason4c_final_safety_audit_passes_metadata_line():
    report = ReasoningFinalSafetyAudit(FinalSafetyAuditConfig.enabled_default()).run().to_dict()

    assert report["enabled"] is True
    assert report["pass_status"] is True
    assert report["safety_flags"]["future_backend_requires_explicit_authorization"] is True
    assert all(finding["status"] == "pass" for finding in report["findings"])
    json.dumps(report)
