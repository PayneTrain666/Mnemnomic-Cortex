import json

from mnemonic_cortex.reasoning_depth import (
    ReasoningStrategyGraph,
    ReasoningStrategyGraphConfig,
    ReasoningStrategyNode,
    ReasoningStrategyEdge,
)


def test_reason3a_strategy_graph_serializes_to_json():
    graph = ReasoningStrategyGraph(ReasoningStrategyGraphConfig.enabled_default())
    q = graph.add_node(ReasoningStrategyNode(node_type="query", label="q", depth_roles=["Z3"]))
    h = graph.add_node(ReasoningStrategyNode(node_type="hypothesis", label="h", depth_roles=["Z4"], confidence=0.7))
    graph.add_edge(ReasoningStrategyEdge(q.node_id, h.node_id, "expands", weight=0.8, confidence=0.7))
    payload = graph.to_dict()

    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1
    json.dumps(payload)
