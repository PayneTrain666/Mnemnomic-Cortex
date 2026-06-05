import json

from mnemonic_cortex.reasoning_depth import BackendThreatModelBuilder, BackendThreatModelConfig


def test_future_backend_threat_model_json_safe_and_blocking():
    report = BackendThreatModelBuilder(BackendThreatModelConfig.enabled_default()).build().to_dict()

    assert report["enabled"] is True
    assert report["threat_count"] >= 5
    assert "credential_scope_review" in report["blocked_until_resolved"]
    assert report["real_store_write_authorized"] is False
    json.dumps(report)
