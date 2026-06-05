from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import hashlib
import json
import uuid


class ReasoningStrategyGraphError(ValueError):
    """Raised when strategy graph construction violates REASON-3A contracts."""


_ALLOWED_NODE_TYPES = {
    "query",
    "hypothesis",
    "evidence",
    "counterfactual",
    "conflict",
    "route",
    "conclusion",
}
_ALLOWED_EDGE_TYPES = {
    "supports",
    "contradicts",
    "expands",
    "tests",
    "routes_to",
    "depends_on",
}


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _safe_jsonable(value.to_dict())
    return str(value)


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(_safe_jsonable(payload), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class ReasoningStrategyGraphConfig:
    """Bounded graph config. Disabled by default."""

    enabled: bool = False
    max_nodes: int = 64
    max_edges: int = 128
    allowed_node_types: Sequence[str] = field(default_factory=lambda: sorted(_ALLOWED_NODE_TYPES))
    allowed_edge_types: Sequence[str] = field(default_factory=lambda: sorted(_ALLOWED_EDGE_TYPES))
    finite_checks: bool = True
    require_json_safe_outputs: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_nodes <= 0 or self.max_nodes > 4096:
            raise ReasoningStrategyGraphError("max_nodes must be in [1,4096]")
        if self.max_edges <= 0 or self.max_edges > 8192:
            raise ReasoningStrategyGraphError("max_edges must be in [1,8192]")
        if not set(self.allowed_node_types).issubset(_ALLOWED_NODE_TYPES):
            raise ReasoningStrategyGraphError("unsupported node type in allowed_node_types")
        if not set(self.allowed_edge_types).issubset(_ALLOWED_EDGE_TYPES):
            raise ReasoningStrategyGraphError("unsupported edge type in allowed_edge_types")
        if not self.no_mutation_by_default:
            raise ReasoningStrategyGraphError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "ReasoningStrategyGraphConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningStrategyGraphConfig":
        return cls(enabled=True)


@dataclass(frozen=True)
class ReasoningStrategyNode:
    """A JSON-safe reasoning strategy node."""

    node_type: str
    label: str
    depth_roles: List[str] = field(default_factory=list)
    confidence: float = 0.5
    disagreement: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.node_type not in _ALLOWED_NODE_TYPES:
            raise ReasoningStrategyGraphError(f"unsupported node_type: {self.node_type}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ReasoningStrategyGraphError("node confidence must be in [0,1]")
        if not (0.0 <= float(self.disagreement) <= 1.0):
            raise ReasoningStrategyGraphError("node disagreement must be in [0,1]")
        if self.node_id is None:
            object.__setattr__(
                self,
                "node_id",
                _stable_id(
                    "node",
                    {
                        "node_type": self.node_type,
                        "label": self.label,
                        "depth_roles": list(self.depth_roles),
                        "evidence_refs": list(self.evidence_refs),
                    },
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "depth_roles": list(self.depth_roles),
            "confidence": float(self.confidence),
            "disagreement": float(self.disagreement),
            "evidence_refs": list(self.evidence_refs),
            "metadata": _safe_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ReasoningStrategyEdge:
    """A JSON-safe reasoning strategy edge."""

    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    edge_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.edge_type not in _ALLOWED_EDGE_TYPES:
            raise ReasoningStrategyGraphError(f"unsupported edge_type: {self.edge_type}")
        if not (0.0 <= float(self.weight) <= 1.0):
            raise ReasoningStrategyGraphError("edge weight must be in [0,1]")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ReasoningStrategyGraphError("edge confidence must be in [0,1]")
        if self.edge_id is None:
            object.__setattr__(
                self,
                "edge_id",
                _stable_id(
                    "edge",
                    {
                        "source": self.source_node_id,
                        "target": self.target_node_id,
                        "edge_type": self.edge_type,
                        "weight": float(self.weight),
                    },
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "weight": float(self.weight),
            "confidence": float(self.confidence),
            "metadata": _safe_jsonable(self.metadata),
        }


@dataclass
class ReasoningStrategyGraph:
    """Bounded non-mutating reasoning strategy graph.

    The graph is an in-memory planning structure only. It does not write to
    WM/MANN/LTM stores, does not activate hidden modules, and exports JSON-safe
    route candidates for downstream planners.
    """

    config: ReasoningStrategyGraphConfig = field(default_factory=ReasoningStrategyGraphConfig.disabled)
    nodes: Dict[str, ReasoningStrategyNode] = field(default_factory=dict)
    edges: Dict[str, ReasoningStrategyEdge] = field(default_factory=dict)
    graph_id: str = field(default_factory=lambda: f"strategy_graph_{uuid.uuid4().hex[:16]}")

    def __post_init__(self) -> None:
        self.config.validate()

    def add_node(self, node: ReasoningStrategyNode) -> ReasoningStrategyNode:
        if len(self.nodes) >= self.config.max_nodes and node.node_id not in self.nodes:
            raise ReasoningStrategyGraphError("max_nodes exceeded")
        if node.node_type not in self.config.allowed_node_types:
            raise ReasoningStrategyGraphError("node type not allowed by config")
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: ReasoningStrategyEdge) -> ReasoningStrategyEdge:
        if len(self.edges) >= self.config.max_edges and edge.edge_id not in self.edges:
            raise ReasoningStrategyGraphError("max_edges exceeded")
        if edge.edge_type not in self.config.allowed_edge_types:
            raise ReasoningStrategyGraphError("edge type not allowed by config")
        if edge.source_node_id not in self.nodes or edge.target_node_id not in self.nodes:
            raise ReasoningStrategyGraphError("edge references unknown node")
        self.edges[edge.edge_id] = edge
        return edge

    def build_default_from_content(self, *, content: str = "", evidence_refs: Optional[List[str]] = None) -> "ReasoningStrategyGraph":
        query = self.add_node(
            ReasoningStrategyNode(
                node_type="query",
                label="input_query",
                depth_roles=["Z3", "Z4"],
                confidence=0.5,
                evidence_refs=evidence_refs or [],
                metadata={"content_present": bool(content)},
            )
        )
        hypothesis = self.add_node(
            ReasoningStrategyNode(
                node_type="hypothesis",
                label=(content[:80] if content else "candidate_hypothesis"),
                depth_roles=["Z4", "Z6"],
                confidence=0.55,
                evidence_refs=evidence_refs or [],
                metadata={"bounded": True},
            )
        )
        self.add_edge(
            ReasoningStrategyEdge(
                source_node_id=query.node_id,
                target_node_id=hypothesis.node_id,
                edge_type="expands",
                weight=0.75,
                confidence=0.55,
            )
        )
        return self

    def route_candidates(self, max_candidates: int = 8) -> List[Dict[str, Any]]:
        if max_candidates <= 0:
            raise ReasoningStrategyGraphError("max_candidates must be positive")
        candidates: List[Dict[str, Any]] = []
        for edge in list(self.edges.values())[:max_candidates]:
            src = self.nodes[edge.source_node_id]
            dst = self.nodes[edge.target_node_id]
            candidates.append(
                {
                    "route_id": _stable_id("route", edge.to_dict()),
                    "source_node_id": src.node_id,
                    "target_node_id": dst.node_id,
                    "edge_type": edge.edge_type,
                    "selected_depth_roles": sorted(set(src.depth_roles + dst.depth_roles)),
                    "confidence": min(src.confidence, dst.confidence, edge.confidence),
                    "disagreement": max(src.disagreement, dst.disagreement),
                    "evidence_refs": sorted(set(src.evidence_refs + dst.evidence_refs)),
                    "metadata": {"graph_id": self.graph_id, "edge_weight": edge.weight},
                }
            )
        return candidates

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "graph_id": self.graph_id,
            "config": {
                "enabled": self.config.enabled,
                "max_nodes": self.config.max_nodes,
                "max_edges": self.config.max_edges,
                "allowed_node_types": list(self.config.allowed_node_types),
                "allowed_edge_types": list(self.config.allowed_edge_types),
                "finite_checks": self.config.finite_checks,
                "require_json_safe_outputs": self.config.require_json_safe_outputs,
                "no_mutation_by_default": self.config.no_mutation_by_default,
            },
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "safety": {
                "external_memory_store_mutation": False,
                "hidden_activation": False,
                "bounded": True,
            },
        }
        if self.config.require_json_safe_outputs:
            json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)


def reasoning_strategy_graph_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_strategy_graph",
        "stage": "REASON-3A",
        "default_enabled": False,
        "bounded_nodes_edges": True,
        "json_safe_export": True,
        "deterministic_ids": True,
        "external_memory_store_mutation": False,
        "hidden_activation": False,
    }
