from mnemonic_cortex.reasoning_depth import ReasoningPersistenceAdapter, ReasoningPersistenceAdapterConfig


def test_reason4a_persistence_adapter_disabled_default():
    adapter = ReasoningPersistenceAdapter(ReasoningPersistenceAdapterConfig.disabled())
    report = adapter.prepare(target_store="strategy_graph", item_kind="graph", items=[{"id": "g1"}])
    payload = report.to_dict()

    assert payload["enabled"] is False
    assert payload["payload"] is None
    assert payload["commit_decision"] is None
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
