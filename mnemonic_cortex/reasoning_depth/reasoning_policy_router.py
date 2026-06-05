from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid

import torch

from .confidence_disagreement_scoring import (
    ConfidenceDisagreementConfig,
    ConfidenceDisagreementReport,
    score_confidence_disagreement,
)
from .depth_route_strategy import (
    DepthRoutePlan,
    DepthRouteStrategyConfig,
    DepthRouteStrategySelector,
    ReasoningTaskMode,
)
from .reasoning_orchestration_trace import _safe_jsonable


class ReasoningPolicyRouterError(ValueError):
    """Raised when the reasoning policy router receives unsafe inputs."""


@dataclass(frozen=True)
class ReasoningPolicyRouterConfig:
    """Optional REASON-2B policy-router config. Disabled by default."""

    enabled: bool = False
    task_mode: str = ReasoningTaskMode.DEFAULT.value
    max_content_chars: int = 4096
    finite_checks: bool = True
    no_mutation_by_default: bool = True
    route_config: DepthRouteStrategyConfig = field(default_factory=DepthRouteStrategyConfig)
    score_config: ConfidenceDisagreementConfig = field(default_factory=ConfidenceDisagreementConfig)

    def validate(self) -> None:
        ReasoningTaskMode(self.task_mode)
        if self.max_content_chars <= 0:
            raise ReasoningPolicyRouterError("max_content_chars must be positive")
        self.route_config.validate()
        self.score_config.validate()

    @classmethod
    def disabled(cls) -> "ReasoningPolicyRouterConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls, task_mode: str = ReasoningTaskMode.DEFAULT.value) -> "ReasoningPolicyRouterConfig":
        return cls(enabled=True, task_mode=task_mode)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "task_mode": self.task_mode,
            "max_content_chars": self.max_content_chars,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
            "route_config": self.route_config.to_dict(),
            "score_config": self.score_config.to_dict(),
        }


@dataclass
class ReasoningPolicyDecision:
    """Router output consumed by ReasoningController."""

    route_plan: DepthRoutePlan
    score_report: ConfidenceDisagreementReport
    enabled: bool
    decision_id: str = field(default_factory=lambda: f"reason_policy_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "enabled": self.enabled,
            "route_plan": self.route_plan.to_dict(),
            "score_report": self.score_report.to_dict(),
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "trace_governance": True,
                "confidence_disagreement_hooks": True,
                "policy_lane_integration": True,
                "write_permission_required_for_commit": True,
                "audit_metadata": True,
            },
            "safety": {
                "non_mutating": True,
                "no_permanent_write": True,
                "bounded_policy_route": True,
            },
        }


class ReasoningPolicyRouter:
    """Optional non-mutating policy router for REASON-2B."""

    def __init__(self, config: Optional[ReasoningPolicyRouterConfig] = None):
        self.config = config or ReasoningPolicyRouterConfig.disabled()
        self.config.validate()
        self.selector = DepthRouteStrategySelector(self.config.route_config)

    def route(
        self,
        query: torch.Tensor,
        *,
        content: str = "",
        task_mode: Optional[str] = None,
        conflict: bool = False,
        uncertainty: float = 0.0,
        support_scores: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningPolicyDecision:
        self._validate_query(query)
        safe_content = (content or "")[: self.config.max_content_chars]
        mode = task_mode or self.config.task_mode

        if support_scores is None:
            scores = query.detach().clone().float().reshape(-1)
            if scores.numel() > self.config.score_config.max_scores:
                scores = scores[: self.config.score_config.max_scores]
        else:
            scores = support_scores

        score_report = score_confidence_disagreement(
            scores,
            config=self.config.score_config,
            metadata={
                "router_enabled": self.config.enabled,
                "query_shape": list(query.shape),
                "content_chars": len(safe_content),
                **(metadata or {}),
            },
        )
        route_plan = self.selector.select(
            task_mode=mode,
            content_hint=safe_content,
            conflict=conflict,
            uncertainty=max(float(uncertainty), float(score_report.disagreement)),
            metadata={
                "router_enabled": self.config.enabled,
                "score_report_id": score_report.report_id,
                **(metadata or {}),
            },
        )
        return ReasoningPolicyDecision(
            route_plan=route_plan,
            score_report=score_report,
            enabled=bool(self.config.enabled),
            metadata={
                "content_truncated": len(content or "") > self.config.max_content_chars,
                "task_mode": mode,
                "conflict": conflict,
                "uncertainty": uncertainty,
            },
        )

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise ReasoningPolicyRouterError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise ReasoningPolicyRouterError("query must be [B,D] or [B,T,D]")
        if query.numel() == 0:
            raise ReasoningPolicyRouterError("query must not be empty")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise ReasoningPolicyRouterError("query contains NaN/Inf")


def reasoning_policy_router_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_policy_router",
        "stage": "REASON-2B",
        "default_enabled": False,
        "non_mutating": True,
        "bounded_route": True,
        "confidence_disagreement_hooks": True,
        "policy_lane_integration": True,
    }
