import json

from mnemonic_cortex.reasoning_depth import ReasoningIntegrationIndex, ReasoningIntegrationIndexConfig


def test_reason4c_integration_index_json_safe():
    report = ReasoningIntegrationIndex(ReasoningIntegrationIndexConfig.enabled_default()).build().to_dict()

    assert report["enabled"] is True
    assert report["entry_count"] >= 16
    assert report["summary"]["stage_4_persistence_metadata"] is True
    assert report["summary"]["permanent_memory_store_mutation"] is False
    json.dumps(report)
