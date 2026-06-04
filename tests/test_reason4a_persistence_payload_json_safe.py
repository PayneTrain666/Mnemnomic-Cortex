import json

from mnemonic_cortex.reasoning_depth import ReasoningPersistenceAdapter, ReasoningPersistenceAdapterConfig


def test_reason4a_persistence_payload_json_safe():
    adapter = ReasoningPersistenceAdapter(ReasoningPersistenceAdapterConfig.enabled_default())
    report = adapter.prepare(
        target_store="reasoning_trace",
        item_kind="trace",
        items=[{"trace_id": "t1", "events": [{"stage": "x"}]}],
        lineage={"stage": "REASON-4A"},
    )
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["payload"]["payload_hash"]
    assert payload["commit_decision"]["approved"] is True
    assert payload["safety_flags"]["real_store_write_performed"] is False
    json.dumps(payload)
