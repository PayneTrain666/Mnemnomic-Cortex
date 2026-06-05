import pytest

from mnemonic_cortex.reasoning_depth import (
    ReasoningStrategyGraph,
    ReasoningStrategyGraphConfig,
    ReasoningStrategyNode,
    ReasoningStrategyGraphError,
)


def test_reason3a_strategy_graph_enforces_max_nodes():
    graph = ReasoningStrategyGraph(ReasoningStrategyGraphConfig(enabled=True, max_nodes=1, max_edges=2))
    graph.add_node(ReasoningStrategyNode(node_type="query", label="q"))
    with pytest.raises(ReasoningStrategyGraphError):
        graph.add_node(ReasoningStrategyNode(node_type="hypothesis", label="h"))
