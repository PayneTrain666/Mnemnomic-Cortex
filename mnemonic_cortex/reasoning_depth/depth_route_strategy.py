from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .reasoning_orchestration_trace import _safe_jsonable


class DepthRouteStrategyError(ValueError):
    """Raised when a route strategy cannot be built safely."""


class ReasoningTaskMode(str, Enum):
    DEFAULT = "default"
    FACT_RECALL = "fact_recall"
    STRUCTURAL_REASONING = "structural_reasoning"
    HYPOTHESIS = "hypothesis"
    EPISODIC_PROJECT = "episodic_project"
    SAFETY_POLICY = "safety_policy"


def normalize_ltm_bank_name(bank_name: str) -> str:
    name = str(bank_name).strip().lower()
    canonical = {
        "hg": "hg_episodic",
        "episodic": "hg_episodic",
        "semantic": "cgmn_semantic",
        "cgmn": "cgmn_semantic",
        "curved": "curved_associative",
        "associative": "curved_associative",
        "spatial": "spatial_topological",
        "spatial_ltm": "spatial_topological",
        "procedural": "procedural_spcp",
        "spcp": "procedural_spcp",
    }.get(name, name)
    allowed = {"hg_episodic", "cgmn_semantic", "curved_associative", "spatial_topological", "procedural_spcp"}
    if canonical not in allowed:
        raise DepthRouteStrategyError(f"unknown preferred_ltm_bank: {bank_name}")
    return canonical


@dataclass(frozen=True)
class DepthRouteStrategyConfig:
    """Deterministic bounded route strategy config."""

    default_depths: tuple[int, ...] = (1, 2, 3, 4)
    max_depths_per_route: int = 5
    max_hops: int = 3
    preferred_ltm_bank: str = "cgmn_semantic"
    include_ltm: bool = True
    include_mann: bool = True
    include_wm: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_depths_per_route <= 0 or self.max_depths_per_route > 8:
            raise DepthRouteStrategyError("max_depths_per_route must be in [1,8]")
        if self.max_hops <= 0 or self.max_hops > 16:
            raise DepthRouteStrategyError("max_hops must be in [1,16]")
        for depth in self.default_depths:
            if depth < 0 or depth > 7:
                raise DepthRouteStrategyError("depth IDs must be in [0,7]")
        normalize_ltm_bank_name(self.preferred_ltm_bank)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_depths": list(self.default_depths),
            "max_depths_per_route": self.max_depths_per_route,
            "max_hops": self.max_hops,
            "preferred_ltm_bank": self.preferred_ltm_bank,
            "include_ltm": self.include_ltm,
            "include_mann": self.include_mann,
            "include_wm": self.include_wm,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class DepthRoutePlan:
    """JSON-safe route plan for one reasoning pass."""

    task_mode: ReasoningTaskMode
    selected_depths: List[int]
    wm_enabled: bool
    mann_enabled: bool
    ltm_enabled: bool
    max_hops: int
    preferred_ltm_bank: str
    route_reason: str
    plan_id: str = field(default_factory=lambda: f"depth_route_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_mode": self.task_mode.value,
            "selected_depths": list(self.selected_depths),
            "wm_enabled": self.wm_enabled,
            "mann_enabled": self.mann_enabled,
            "ltm_enabled": self.ltm_enabled,
            "max_hops": self.max_hops,
            "preferred_ltm_bank": self.preferred_ltm_bank,
            "route_reason": self.route_reason,
            "metadata": _safe_jsonable(self.metadata),
            "safety": {
                "bounded_depths": True,
                "bounded_hops": True,
                "non_mutating": True,
                "no_permanent_write": True,
            },
        }


class DepthRouteStrategySelector:
    """Small deterministic strategy selector without unbounded search."""

    def __init__(self, config: Optional[DepthRouteStrategyConfig] = None):
        self.config = config or DepthRouteStrategyConfig()
        self.config.validate()

    def select(
        self,
        *,
        task_mode: ReasoningTaskMode | str = ReasoningTaskMode.DEFAULT,
        content_hint: str = "",
        conflict: bool = False,
        uncertainty: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DepthRoutePlan:
        mode = ReasoningTaskMode(task_mode) if not isinstance(task_mode, ReasoningTaskMode) else task_mode
        content_l = (content_hint or "").lower()

        if conflict:
            depths = [1, 2, 4, 6, 7]
            reason = "conflict flag routes through invariant/structure/reasoning/hypothesis/volatile depths"
        elif mode == ReasoningTaskMode.FACT_RECALL:
            depths = [1, 3, 5]
            reason = "fact recall routes through semantic/contextual/episode depths"
        elif mode == ReasoningTaskMode.STRUCTURAL_REASONING:
            depths = [2, 4, 1]
            reason = "structural reasoning routes through relation/transform/invariant depths"
        elif mode == ReasoningTaskMode.HYPOTHESIS:
            depths = [6, 4, 2, 7]
            reason = "hypothesis mode routes through hypothesis/reasoning/structure/volatile depths"
        elif mode == ReasoningTaskMode.EPISODIC_PROJECT or any(term in content_l for term in ("project", "chat", "episode")):
            depths = [5, 3, 1, 7]
            reason = "project/episode content routes through episode/context/invariant/volatile depths"
        elif mode == ReasoningTaskMode.SAFETY_POLICY:
            depths = [0, 1, 4, 7]
            reason = "safety/policy content routes through identity/invariant/reasoning/volatile depths"
        else:
            depths = list(self.config.default_depths)
            reason = "default bounded route"

        if uncertainty > 0.55 and 6 not in depths:
            depths.append(6)
            reason += "; uncertainty added hypothesis depth"

        selected = []
        for depth in depths:
            if depth not in selected:
                selected.append(depth)
            if len(selected) >= self.config.max_depths_per_route:
                break

        max_hops = min(self.config.max_hops, 1 + max(0, len(selected) // 2))
        return DepthRoutePlan(
            task_mode=mode,
            selected_depths=selected,
            wm_enabled=self.config.include_wm,
            mann_enabled=self.config.include_mann,
            ltm_enabled=self.config.include_ltm,
            max_hops=max_hops,
            preferred_ltm_bank=self._preferred_ltm_bank(mode, content_l),
            route_reason=reason,
            metadata=metadata or {},
        )

    def _preferred_ltm_bank(self, mode: ReasoningTaskMode, content_l: str) -> str:
        configured = normalize_ltm_bank_name(self.config.preferred_ltm_bank)
        if configured != "cgmn_semantic":
            return configured
        if any(term in content_l for term in ("spatial", "location", "map", "pose", "quaternion")):
            return "spatial_topological"
        if mode == ReasoningTaskMode.STRUCTURAL_REASONING:
            return "curved_associative"
        if mode == ReasoningTaskMode.HYPOTHESIS:
            return "procedural_spcp"
        if mode == ReasoningTaskMode.EPISODIC_PROJECT or any(term in content_l for term in ("project", "chat", "episode")):
            return "hg_episodic"
        return configured

def depth_route_strategy_contract() -> Dict[str, Any]:
    return {
        "module": "depth_route_strategy",
        "stage": "REASON-2B",
        "bounded_depth_route": True,
        "bounded_hops": True,
        "non_mutating": True,
        "canonical_ltm_banks": ["hg_episodic", "cgmn_semantic", "curved_associative", "spatial_topological", "procedural_spcp"],
        "task_mode_aware_ltm_bank_selection": True,
        "default_depth_count_max": 8,
    }
