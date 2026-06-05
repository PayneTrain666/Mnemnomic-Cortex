from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

import torch

from .reasoning_strategy_graph import (
    ReasoningStrategyGraph,
    ReasoningStrategyGraphConfig,
    ReasoningStrategyNode,
    ReasoningStrategyGraphError,
    _safe_jsonable,
    _stable_id,
)
from .evidence_guided_route_expander import (
    EvidenceGuidedRouteExpander,
    EvidenceGuidedRouteExpanderConfig,
    EvidenceGuidedRouteExpansionReport,
)


class MultiPassThoughtPlannerError(ValueError):
    """Raised when multi-pass planning violates safety or boundedness."""


@dataclass(frozen=True)
class MultiPassThoughtPlannerConfig:
    """Bounded multi-pass planner config. Disabled by default."""

    enabled: bool = False
    max_passes: int = 3
    max_routes_per_pass: int = 4
    max_graph_nodes: int = 64
    max_graph_edges: int = 128
    stop_on_conflict: bool = True
    stop_on_low_confidence: bool = False
    low_confidence_floor: float = 0.2
    finite_checks: bool = True
    no_mutation_by_default: bool = True
    route_expander_config: EvidenceGuidedRouteExpanderConfig = field(default_factory=EvidenceGuidedRouteExpanderConfig.disabled)

    def validate(self) -> None:
        if self.max_passes <= 0 or self.max_passes > 32:
            raise MultiPassThoughtPlannerError("max_passes must be in [1,32]")
        if self.max_routes_per_pass <= 0 or self.max_routes_per_pass > 64:
            raise MultiPassThoughtPlannerError("max_routes_per_pass must be in [1,64]")
        if self.max_graph_nodes <= 0 or self.max_graph_edges <= 0:
            raise MultiPassThoughtPlannerError("graph bounds must be positive")
        if not (0.0 <= self.low_confidence_floor <= 1.0):
            raise MultiPassThoughtPlannerError("low_confidence_floor must be in [0,1]")
        if not self.no_mutation_by_default:
            raise MultiPassThoughtPlannerError("no_mutation_by_default must remain true")
        self.route_expander_config.validate()

    @classmethod
    def disabled(cls) -> "MultiPassThoughtPlannerConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "MultiPassThoughtPlannerConfig":
        return cls(enabled=True, route_expander_config=EvidenceGuidedRouteExpanderConfig.enabled_default())


@dataclass
class ThoughtPlanPass:
    """One bounded planning pass."""

    pass_id: str
    route_id: str
    selected_strategy_nodes: List[str]
    selected_depth_roles: List[str]
    confidence: float
    disagreement: float
    evidence_support: float
    conflict_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "pass_id": self.pass_id,
                "route_id": self.route_id,
                "selected_strategy_nodes": list(self.selected_strategy_nodes),
                "selected_depth_roles": list(self.selected_depth_roles),
                "confidence": float(self.confidence),
                "disagreement": float(self.disagreement),
                "evidence_support": float(self.evidence_support),
                "conflict_status": self.conflict_status,
                "metadata": self.metadata,
            }
        )


@dataclass
class ThoughtPlanReport:
    """JSON-safe planner report."""

    enabled: bool
    passes: List[ThoughtPlanPass]
    graph: Optional[ReasoningStrategyGraph] = None
    route_expansion: Optional[EvidenceGuidedRouteExpansionReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"thought_plan_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "report_id": self.report_id,
                "enabled": self.enabled,
                "passes": [item.to_dict() for item in self.passes],
                "graph": self.graph.to_dict() if self.graph is not None else None,
                "route_expansion": self.route_expansion.to_dict() if self.route_expansion is not None else None,
                "metadata": self.metadata,
                "paamax_metadata": {
                    "trace_governance": True,
                    "confidence_hook": True,
                    "disagreement_hook": True,
                    "conflict_hook": True,
                    "quarantine_hook": True,
                    "policy_lane_integration": True,
                    "write_permission_required_for_commit": True,
                },
                "safety": {
                    "permanent_memory_store_mutation": False,
                    "model_weight_mutation": False,
                    "optimizer_mutation": False,
                    "bounded": True,
                    "real_ablation_execution": False,
                },
            }
        )


class MultiPassThoughtPlanner:
    """Optional bounded multi-pass thought planner.

    The planner is metadata/control-plane only. It does not write to memory
    stores, does not mutate input tensors, and remains disabled unless explicit
    config enables it.
    """

    def __init__(self, config: Optional[MultiPassThoughtPlannerConfig] = None):
        self.config = config or MultiPassThoughtPlannerConfig.disabled()
        self.config.validate()
        self.expander = EvidenceGuidedRouteExpander(self.config.route_expander_config)

    def plan(
        self,
        query: torch.Tensor,
        *,
        content: str = "",
        evidence_report: Optional[Any] = None,
        graph: Optional[ReasoningStrategyGraph] = None,
    ) -> ThoughtPlanReport:
        if not isinstance(query, torch.Tensor) or query.dim() not in (2, 3):
            raise MultiPassThoughtPlannerError("query must be [B,D] or [B,T,D]")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise MultiPassThoughtPlannerError("query contains NaN/Inf")

        if not self.config.enabled:
            return ThoughtPlanReport(
                enabled=False,
                passes=[],
                metadata={"reason": "planner disabled", "query_shape": list(query.shape)},
            )

        before = query.clone()
        working_graph = graph or ReasoningStrategyGraph(
            ReasoningStrategyGraphConfig(
                enabled=True,
                max_nodes=self.config.max_graph_nodes,
                max_edges=self.config.max_graph_edges,
            )
        )
        if not working_graph.nodes:
            evidence_refs = self._evidence_refs(evidence_report)
            working_graph.build_default_from_content(content=content, evidence_refs=evidence_refs)
            if content:
                working_graph.add_node(
                    ReasoningStrategyNode(
                        node_type="evidence",
                        label="evidence_context",
                        depth_roles=["Z1", "Z3"],
                        confidence=0.65,
                        evidence_refs=evidence_refs,
                        metadata={"content_len": len(content)},
                    )
                )

        routes = working_graph.route_candidates(max_candidates=self.config.max_passes * self.config.max_routes_per_pass)
        route_expansion = self.expander.expand(routes, evidence_report=evidence_report)

        selected = route_expansion.candidates if route_expansion.enabled else []
        if not selected:
            selected = []
            for raw in routes[: self.config.max_passes * self.config.max_routes_per_pass]:
                selected.append(
                    type(
                        "_Route",
                        (),
                        {
                            "route_id": raw["route_id"],
                            "selected_depth_roles": raw.get("selected_depth_roles", []),
                            "confidence": raw.get("confidence", 0.5),
                            "disagreement": raw.get("disagreement", 0.0),
                            "evidence_support": 0.0,
                            "conflict_prone": raw.get("disagreement", 0.0) > 0.75,
                            "unsupported": False,
                        },
                    )()
                )

        passes: List[ThoughtPlanPass] = []
        for index, route in enumerate(selected[: self.config.max_passes]):
            if self.config.stop_on_conflict and bool(getattr(route, "conflict_prone", False)):
                conflict_status = "stopped_on_conflict"
            elif self.config.stop_on_low_confidence and float(getattr(route, "confidence", 0.0)) < self.config.low_confidence_floor:
                conflict_status = "stopped_on_low_confidence"
            else:
                conflict_status = "clear"

            passes.append(
                ThoughtPlanPass(
                    pass_id=f"pass_{index}",
                    route_id=str(getattr(route, "route_id")),
                    selected_strategy_nodes=[],
                    selected_depth_roles=list(getattr(route, "selected_depth_roles", [])),
                    confidence=float(getattr(route, "confidence", 0.5)),
                    disagreement=float(getattr(route, "disagreement", 0.0)),
                    evidence_support=float(getattr(route, "evidence_support", 0.0)),
                    conflict_status=conflict_status,
                    metadata={"unsupported": bool(getattr(route, "unsupported", False))},
                )
            )
            if conflict_status.startswith("stopped"):
                break

        if not torch.equal(query, before):
            raise MultiPassThoughtPlannerError("query tensor was mutated during planning")

        report = ThoughtPlanReport(
            enabled=True,
            passes=passes,
            graph=working_graph,
            route_expansion=route_expansion,
            metadata={
                "max_passes": self.config.max_passes,
                "max_routes_per_pass": self.config.max_routes_per_pass,
                "query_shape": list(query.shape),
            },
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _evidence_refs(evidence_report: Optional[Any]) -> List[str]:
        if evidence_report is None:
            return []
        if hasattr(evidence_report, "to_dict"):
            payload = evidence_report.to_dict()
        elif isinstance(evidence_report, dict):
            payload = evidence_report
        else:
            return []
        refs = []
        for unit in payload.get("evidence_units", []):
            if isinstance(unit, dict):
                refs.append(str(unit.get("unit_id", unit.get("id", ""))))
        return [ref for ref in refs if ref]


def multi_pass_thought_planner_contract() -> Dict[str, Any]:
    return {
        "module": "multi_pass_thought_planner",
        "stage": "REASON-3A",
        "default_enabled": False,
        "bounded_passes": True,
        "json_safe_report": True,
        "input_tensor_mutation": False,
        "permanent_memory_store_mutation": False,
        "paamax_metadata": True,
    }
