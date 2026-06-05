from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import uuid

from .planner_evaluation import _safe_jsonable
from .planner_failure_classifier import PlannerFailureRecord, PlannerFailureSeverity


class PlannerRemediationGuidanceError(ValueError):
    """Raised when remediation guidance violates recommendation-only safety."""


_BLOCKED_UNSAFE_ACTIONS = [
    "automatic_memory_store_write",
    "automatic_model_mutation",
    "automatic_optimizer_mutation",
    "automatic_policy_activation",
    "automatic_component_activation",
    "real_ablation_execution",
    "destructive_replacement",
    "automatic_patch_application",
]


@dataclass(frozen=True)
class PlannerRemediationGuidanceConfig:
    """Recommendation-only guidance config. Disabled by default."""

    enabled: bool = False
    max_guidance_items: int = 32
    recommendation_only: bool = True
    allow_patch_application: bool = False
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_guidance_items <= 0 or self.max_guidance_items > 1024:
            raise PlannerRemediationGuidanceError("max_guidance_items must be in [1,1024]")
        if not self.recommendation_only:
            raise PlannerRemediationGuidanceError("recommendation_only must remain true")
        if self.allow_patch_application:
            raise PlannerRemediationGuidanceError("allow_patch_application must remain false")
        if not self.no_mutation_by_default:
            raise PlannerRemediationGuidanceError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PlannerRemediationGuidanceConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PlannerRemediationGuidanceConfig":
        return cls(enabled=True)


def _stable_guidance_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(_safe_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return f"planner_guidance_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class PlannerRemediationGuidanceItem:
    """A JSON-safe, recommendation-only remediation guidance item."""

    related_failure_id: str
    recommended_action: str
    recommended_owner: str
    patch_category: str
    required_evidence: List[str] = field(default_factory=list)
    blocked_unsafe_actions: List[str] = field(default_factory=lambda: list(_BLOCKED_UNSAFE_ACTIONS))
    priority: str = "medium"
    downstream_impact: str = ""
    guidance_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.guidance_id is None:
            object.__setattr__(
                self,
                "guidance_id",
                _stable_guidance_id(
                    {
                        "related_failure_id": self.related_failure_id,
                        "recommended_action": self.recommended_action,
                        "patch_category": self.patch_category,
                        "priority": self.priority,
                    }
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "guidance_id": self.guidance_id,
                "related_failure_id": self.related_failure_id,
                "recommended_action": self.recommended_action,
                "recommended_owner": self.recommended_owner,
                "patch_category": self.patch_category,
                "required_evidence": list(self.required_evidence),
                "blocked_unsafe_actions": list(self.blocked_unsafe_actions),
                "priority": self.priority,
                "downstream_impact": self.downstream_impact,
            }
        )


@dataclass
class PlannerRemediationGuidanceReport:
    """JSON-safe recommendation-only guidance report."""

    enabled: bool
    items: List[PlannerRemediationGuidanceItem]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"planner_guidance_report_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "items": [item.to_dict() for item in self.items],
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
                "policy_lane_integration": True,
                "quarantine_hook": True,
            },
        }
        json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)


class PlannerRemediationGuidance:
    """Produces recommendation-only remediation guidance from failure records."""

    def __init__(self, config: Optional[PlannerRemediationGuidanceConfig] = None):
        self.config = config or PlannerRemediationGuidanceConfig.disabled()
        self.config.validate()

    def recommend(self, failure_records: Iterable[Any], *, lineage: Optional[Dict[str, Any]] = None) -> PlannerRemediationGuidanceReport:
        if not self.config.enabled:
            return PlannerRemediationGuidanceReport(
                enabled=False,
                items=[],
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        items: List[PlannerRemediationGuidanceItem] = []
        for raw in list(failure_records)[: self.config.max_guidance_items]:
            record = self._as_record_dict(raw)
            severity = str(record.get("severity", "medium"))
            family = str(record.get("family", "unknown"))
            priority = "high" if severity in {PlannerFailureSeverity.BLOCKER.value, PlannerFailureSeverity.HIGH.value} else severity
            action = self._action_for_family(family)
            items.append(
                PlannerRemediationGuidanceItem(
                    related_failure_id=str(record.get("failure_id", "unknown_failure")),
                    recommended_action=action,
                    recommended_owner="planner_quality_hardening_stage",
                    patch_category="recommendation_only",
                    required_evidence=[
                        "planner trace payload",
                        "evaluation report",
                        "failure record",
                        "boundedness and no-mutation audit",
                    ],
                    priority=priority,
                    downstream_impact=f"Improves planner quality for failure family: {family}",
                )
            )

        report = PlannerRemediationGuidanceReport(
            enabled=True,
            items=items,
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _as_record_dict(record: Any) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            return record.to_dict()
        if isinstance(record, dict):
            return record
        raise PlannerRemediationGuidanceError("failure record must be record-like or dict")

    @staticmethod
    def _action_for_family(family: str) -> str:
        return {
            "empty_plan": "Add route construction evidence and verify graph route candidate extraction.",
            "low_confidence": "Require stronger evidence support or demote route below consolidation threshold.",
            "high_disagreement": "Send route through conflict-aware review and quarantine if unresolved.",
            "low_evidence_support": "Collect more evidence units before accepting planner route.",
            "conflict_prone_route": "Block consolidation and require contradiction-resolution pass.",
            "unsupported_route": "Demote unsupported route or attach supporting evidence references.",
            "unsafe_mutation_signal": "Fail closed, block commit path, and audit write-permission plumbing.",
            "excessive_pass_count": "Lower max_passes or add bounded early-stop criteria.",
            "excessive_route_count": "Lower max_route_candidates or filter routes before expansion.",
        }.get(family, "Review planner trace, evaluation report, and route metadata.")

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "recommendation_only": True,
            "allow_patch_application": False,
            "automatic_memory_store_write": False,
            "automatic_model_mutation": False,
            "automatic_optimizer_mutation": False,
            "automatic_policy_activation": False,
            "automatic_component_activation": False,
            "real_ablation_execution": False,
            "destructive_replacement": False,
            "blocked_unsafe_actions": list(_BLOCKED_UNSAFE_ACTIONS),
        }


def planner_remediation_guidance_contract() -> Dict[str, Any]:
    return {
        "module": "planner_remediation_guidance",
        "stage": "REASON-3B",
        "default_enabled": False,
        "recommendation_only": True,
        "allow_patch_application": False,
        "blocked_unsafe_actions": list(_BLOCKED_UNSAFE_ACTIONS),
        "json_safe_report": True,
        "permanent_memory_store_mutation": False,
    }
