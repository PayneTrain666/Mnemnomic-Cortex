from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

from .planner_evaluation import _safe_jsonable


class StrategyGraphPersistenceReadinessError(ValueError):
    """Raised when persistence-readiness metadata is unsafe."""


@dataclass(frozen=True)
class StrategyGraphPersistenceReadinessConfig:
    """Persistence readiness config.

    This is metadata-only. It does not persist strategy graphs and does not
    mutate stores.
    """

    enabled: bool = False
    require_graph_json_safe: bool = True
    require_node_ids: bool = True
    require_edge_ids: bool = True
    require_lineage: bool = True
    allow_automatic_persistence: bool = False
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.allow_automatic_persistence:
            raise StrategyGraphPersistenceReadinessError("automatic persistence is forbidden")
        if not self.no_mutation_by_default:
            raise StrategyGraphPersistenceReadinessError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "StrategyGraphPersistenceReadinessConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "StrategyGraphPersistenceReadinessConfig":
        return cls(enabled=True)


@dataclass
class StrategyGraphPersistenceReadinessReport:
    """JSON-safe metadata-only persistence readiness report."""

    enabled: bool
    ready: bool
    missing_requirements: List[str] = field(default_factory=list)
    graph_summary: Dict[str, Any] = field(default_factory=dict)
    recommended_next_steps: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"strategy_graph_persistence_readiness_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "ready": bool(self.ready),
            "missing_requirements": list(self.missing_requirements),
            "graph_summary": _safe_jsonable(self.graph_summary),
            "recommended_next_steps": list(self.recommended_next_steps),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "audit_metadata": True,
                "trace_governance": True,
                "write_permission_required_for_commit": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class StrategyGraphPersistenceReadinessChecker:
    """Checks strategy graph persistence readiness without persisting anything."""

    def __init__(self, config: Optional[StrategyGraphPersistenceReadinessConfig] = None):
        self.config = config or StrategyGraphPersistenceReadinessConfig.disabled()
        self.config.validate()

    def check(self, graph: Optional[Any], *, lineage: Optional[Dict[str, Any]] = None) -> StrategyGraphPersistenceReadinessReport:
        if not self.config.enabled:
            return StrategyGraphPersistenceReadinessReport(
                enabled=False,
                ready=False,
                missing_requirements=["checker_disabled"],
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        payload = self._as_graph_payload(graph)
        missing: List[str] = []
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        if self.config.require_graph_json_safe:
            try:
                json.dumps(_safe_jsonable(payload), sort_keys=True)
            except Exception:
                missing.append("graph_json_safe")
        if self.config.require_node_ids:
            if not isinstance(nodes, list) or any(not isinstance(n, dict) or not n.get("node_id") for n in nodes):
                missing.append("node_ids")
        if self.config.require_edge_ids:
            if not isinstance(edges, list) or any(not isinstance(e, dict) or not e.get("edge_id") for e in edges):
                missing.append("edge_ids")
        if self.config.require_lineage and not lineage:
            missing.append("lineage")

        ready = not missing
        next_steps = []
        if "lineage" in missing:
            next_steps.append("Attach source pack, stage ID, and graph owner lineage before persistence.")
        if "node_ids" in missing:
            next_steps.append("Ensure every strategy graph node has deterministic node_id.")
        if "edge_ids" in missing:
            next_steps.append("Ensure every strategy graph edge has deterministic edge_id.")
        if not next_steps:
            next_steps.append("Proceed to explicit persistence adapter design; do not enable automatic writes.")

        return StrategyGraphPersistenceReadinessReport(
            enabled=True,
            ready=ready,
            missing_requirements=missing,
            graph_summary={
                "node_count": len(nodes) if isinstance(nodes, list) else 0,
                "edge_count": len(edges) if isinstance(edges, list) else 0,
                "has_route_candidates": bool(payload.get("route_candidates", [])),
            },
            recommended_next_steps=next_steps,
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )

    @staticmethod
    def _as_graph_payload(graph: Optional[Any]) -> Dict[str, Any]:
        if graph is None:
            return {}
        if hasattr(graph, "to_dict"):
            return _safe_jsonable(graph.to_dict())
        if isinstance(graph, dict):
            return _safe_jsonable(graph)
        raise StrategyGraphPersistenceReadinessError("graph must be graph-like or dict")

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "metadata_only": True,
            "automatic_persistence": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
        }


def strategy_graph_persistence_readiness_contract() -> Dict[str, Any]:
    return {
        "module": "strategy_graph_persistence_readiness",
        "stage": "REASON-3C",
        "default_enabled": False,
        "metadata_only": True,
        "automatic_persistence": False,
        "permanent_memory_store_mutation": False,
        "json_safe_report": True,
    }
