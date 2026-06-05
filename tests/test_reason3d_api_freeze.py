import json

from mnemonic_cortex.reasoning_depth import ReasoningAPIFreeze, ReasoningAPIFreezeConfig


def test_reason3d_api_freeze_metadata():
    report = ReasoningAPIFreeze(ReasoningAPIFreezeConfig.enabled_default()).freeze(lineage={"stage": "test"})
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert "ReasoningControllerAPI" in payload["frozen_symbols"]
    assert "allow_controller_planner_integration" in payload["config_fields"]
    assert payload["contract_hash"]
    assert payload["safety_flags"]["api_breaking_changes_allowed"] is False
    json.dumps(payload)
