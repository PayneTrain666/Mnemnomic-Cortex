import json

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason2d_config_round_trip_builds_controller():
    cfg = ReasoningControllerAPIConfig(
        enabled=True,
        key_dim=8,
        value_dim=8,
        slot_count=4,
        allow_evidence_reasoning=True,
    )
    payload = cfg.to_dict()
    rebuilt = ReasoningControllerAPIConfig.from_dict(json.loads(json.dumps(payload)))
    api = ReasoningControllerAPI(rebuilt)

    assert api.config.to_dict() == payload
    assert api.contract_summary()["canonical_slot_prefix_compatibility"] == "reason2a."
