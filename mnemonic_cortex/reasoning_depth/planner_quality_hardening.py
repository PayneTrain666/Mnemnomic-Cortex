from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import uuid

from .planner_evaluation import PlannerEvaluationReport, _safe_jsonable
from .planner_failure_classifier import PlannerFailureRecord, PlannerFailureSeverity


class PlannerQualityHardeningError(ValueError):
    """Raised when planner quality hardening violates safety contracts."""


def _stable_quality_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(_safe_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class PlannerQualityHardeningConfig:
    """Planner quality hardening config.

    Disabled by default. This module evaluates planner quality and emits
    recommendation-only hardening actions. It does not mutate memory stores,
    model weights, optimizer state, policy state, or controller state.
    """

    enabled: bool = False
    min_overall_score: float = 0.45
    blocker_fail_closed: bool = True
    max_hardening_actions: int = 32
    recommendation_only: bool = True
    allow_runtime_patch_application: bool = False
    require_json_safe_outputs: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not (0.0 <= float(self.min_overall_score) <= 1.0):
            raise PlannerQualityHardeningError("min_overall_score must be in [0,1]")
        if self.max_hardening_actions <= 0 or self.max_hardening_actions > 1024:
            raise PlannerQualityHardeningError("max_hardening_actions must be in [1,1024]")
        if not self.recommendation_only:
            raise PlannerQualityHardeningError("recommendation_only must remain true")
        if self.allow_runtime_patch_application:
            raise PlannerQualityHardeningError("runtime patch application is forbidden")
        if not self.no_mutation_by_default:
            raise PlannerQualityHardeningError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PlannerQualityHardeningConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PlannerQualityHardeningConfig":
        return cls(enabled=True)


@dataclass(frozen=True)
class PlannerQualityAction:
    """Recommendation-only planner quality hardening action."""

    action_type: str
    reason: str
    priority: str = "medium"
    related_failure_ids: List[str] = field(default_factory=list)
    recommended_change: str = ""
    required_evidence: List[str] = field(default_factory=list)
    blocked_unsafe_actions: List[str] = field(default_factory=lambda: [
        "automatic_memory_store_write",
        "automatic_model_mutation",
        "automatic_optimizer_mutation",
        "automatic_policy_activation",
        "automatic_component_activation",
        "automatic_patch_application",
        "real_ablation_execution",
        "destructive_replacement",
    ])
    action_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action_id is None:
            object.__setattr__(
                self,
                "action_id",
                _stable_quality_id(
                    "planner_quality_action",
                    {
                        "action_type": self.action_type,
                        "reason": self.reason,
                        "priority": self.priority,
                        "related_failure_ids": list(self.related_failure_ids),
                    },
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "action_id": self.action_id,
                "action_type": self.action_type,
                "reason": self.reason,
                "priority": self.priority,
                "related_failure_ids": list(self.related_failure_ids),
                "recommended_change": self.recommended_change,
                "required_evidence": list(self.required_evidence),
                "blocked_unsafe_actions": list(self.blocked_unsafe_actions),
            }
        )


@dataclass
class PlannerQualityHardeningReport:
    """JSON-safe planner quality hardening report."""

    enabled: bool
    quality_level: str
    pass_gate: bool
    actions: List[PlannerQualityAction] = field(default_factory=list)
    evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    failure_summary: Dict[str, Any] = field(default_factory=dict)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"planner_quality_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "quality_level": self.quality_level,
            "pass_gate": bool(self.pass_gate),
            "actions": [action.to_dict() for action in self.actions],
            "evaluation_summary": _safe_jsonable(self.evaluation_summary),
            "failure_summary": _safe_jsonable(self.failure_summary),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "confidence_hook": True,
                "disagreement_hook": True,
                "conflict_hook": True,
                "quarantine_hook": True,
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
                "policy_lane_integration": True,
            },
        }
        json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)


class PlannerQualityHardener:
    """Recommendation-only planner quality hardener."""

    def __init__(self, config: Optional[PlannerQualityHardeningConfig] = None):
        self.config = config or PlannerQualityHardeningConfig.disabled()
        self.config.validate()

    def harden(
        self,
        *,
        evaluation_report: Optional[Any] = None,
        failure_records: Optional[Iterable[Any]] = None,
        remediation_report: Optional[Any] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> PlannerQualityHardeningReport:
        if not self.config.enabled:
            return PlannerQualityHardeningReport(
                enabled=False,
                quality_level="disabled",
                pass_gate=False,
                actions=[],
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        evaluation = self._as_dict(evaluation_report)
        score = evaluation.get("score", {}) if isinstance(evaluation.get("score", {}), dict) else {}
        failures = [self._as_dict(item) for item in list(failure_records or [])]
        remediation = self._as_dict(remediation_report)

        blocker_failures = [item for item in failures if str(item.get("severity", "")).lower() == PlannerFailureSeverity.BLOCKER.value]
        high_failures = [item for item in failures if str(item.get("severity", "")).lower() == PlannerFailureSeverity.HIGH.value]
        overall = float(score.get("overall_score", 0.0))
        pass_gate = overall >= self.config.min_overall_score and not blocker_failures
        if self.config.blocker_fail_closed and blocker_failures:
            pass_gate = False

        quality_level = self._quality_level(overall, blocker_count=len(blocker_failures), high_count=len(high_failures))
        actions = self._build_actions(score, failures, remediation)[: self.config.max_hardening_actions]

        report = PlannerQualityHardeningReport(
            enabled=True,
            quality_level=quality_level,
            pass_gate=pass_gate,
            actions=actions,
            evaluation_summary={
                "overall_score": overall,
                "pass_count": int(score.get("pass_count", 0)),
                "route_count": int(score.get("route_count", 0)),
                "mean_confidence": float(score.get("mean_confidence", 0.0)),
                "max_disagreement": float(score.get("max_disagreement", 0.0)),
                "mean_evidence_support": float(score.get("mean_evidence_support", 0.0)),
            },
            failure_summary={
                "failure_count": len(failures),
                "blocker_count": len(blocker_failures),
                "high_count": len(high_failures),
                "families": [str(item.get("family", "unknown")) for item in failures],
            },
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        if self.config.require_json_safe_outputs:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
        elif isinstance(value, dict):
            payload = value
        else:
            return {"raw": str(value)}
        return _safe_jsonable(payload)

    @staticmethod
    def _quality_level(overall: float, *, blocker_count: int, high_count: int) -> str:
        if blocker_count:
            return "blocked"
        if high_count:
            return "needs_hardening"
        if overall >= 0.80:
            return "strong"
        if overall >= 0.55:
            return "acceptable"
        return "weak"

    @staticmethod
    def _build_actions(score: Dict[str, Any], failures: List[Dict[str, Any]], remediation: Dict[str, Any]) -> List[PlannerQualityAction]:
        actions: List[PlannerQualityAction] = []
        related_ids = [str(item.get("failure_id", "")) for item in failures if item.get("failure_id")]
        if float(score.get("mean_confidence", 1.0)) < 0.35:
            actions.append(PlannerQualityAction(
                action_type="raise_confidence",
                reason="mean confidence below target",
                priority="high",
                related_failure_ids=related_ids,
                recommended_change="Tighten route selection and require stronger evidence before route acceptance.",
                required_evidence=["evaluation score", "planner trace", "evidence support report"],
            ))
        if float(score.get("max_disagreement", 0.0)) > 0.75:
            actions.append(PlannerQualityAction(
                action_type="reduce_disagreement",
                reason="max disagreement above ceiling",
                priority="high",
                related_failure_ids=related_ids,
                recommended_change="Route disputed plan branches through conflict-aware quarantine before consolidation.",
                required_evidence=["conflict records", "counter-evidence", "planner trace"],
            ))
        if int(score.get("conflict_route_count", 0)) > 0:
            actions.append(PlannerQualityAction(
                action_type="quarantine_conflict_routes",
                reason="conflict-prone routes detected",
                priority="high",
                related_failure_ids=related_ids,
                recommended_change="Flag conflicting routes as quarantine candidates and block write gates.",
                required_evidence=["conflict-aware consolidation report", "route expansion report"],
            ))
        if int(score.get("unsupported_route_count", 0)) > 0:
            actions.append(PlannerQualityAction(
                action_type="demote_unsupported_routes",
                reason="unsupported routes detected",
                priority="medium",
                related_failure_ids=related_ids,
                recommended_change="Demote unsupported routes until evidence references are attached.",
                required_evidence=["evidence references", "route candidate list"],
            ))
        for item in remediation.get("items", []) if isinstance(remediation.get("items", []), list) else []:
            actions.append(PlannerQualityAction(
                action_type="adopt_recommendation",
                reason=str(item.get("recommended_action", "planner remediation recommendation")),
                priority=str(item.get("priority", "medium")),
                related_failure_ids=[str(item.get("related_failure_id", ""))],
                recommended_change=str(item.get("recommended_action", "")),
                required_evidence=list(item.get("required_evidence", [])) if isinstance(item.get("required_evidence", []), list) else [],
            ))
        if not actions:
            actions.append(PlannerQualityAction(
                action_type="monitor",
                reason="no hard blocker detected",
                priority="low",
                recommended_change="Continue bounded evaluation and keep planner opt-in.",
                required_evidence=["periodic planner evaluation report"],
            ))
        return actions

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "recommendation_only": True,
            "automatic_patch_application": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "automatic_policy_activation": False,
            "automatic_component_activation": False,
            "real_ablation_execution": False,
            "destructive_replacement": False,
        }


def planner_quality_hardening_contract() -> Dict[str, Any]:
    return {
        "module": "planner_quality_hardening",
        "stage": "REASON-3C",
        "default_enabled": False,
        "recommendation_only": True,
        "automatic_patch_application": False,
        "permanent_memory_store_mutation": False,
        "json_safe_report": True,
        "paamax_metadata": True,
    }
