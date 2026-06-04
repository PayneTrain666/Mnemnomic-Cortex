import json

from mnemonic_cortex.reasoning_depth import (
    ReasoningStrategyGraph,
    ReasoningStrategyGraphConfig,
    ReasoningStrategyNode,
    StrategyGraphPersistenceReadinessChecker,
    StrategyGraphPersistenceReadinessConfig,
)


def test_reason3c_strategy_graph_persistence_readiness_metadata_only():
    graph = ReasoningStrategyGraph(ReasoningStrategyGraphConfig.enabled_default())
    graph.add_node(ReasoningStrategyNode(node_type="query", label="q"))
    checker = StrategyGraphPersistenceReadinessChecker(StrategyGraphPersistenceReadinessConfig.enabled_default())
    report = checker.check(graph, lineage={"stage": "REASON-3C"})
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["ready"] is True
    assert payload["safety_flags"]["automatic_persistence"] is False
    assert payload["safety_flags"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
