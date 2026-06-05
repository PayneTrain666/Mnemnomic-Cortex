from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
import uuid

import torch

from .multi_pass_thought_planner import MultiPassThoughtPlanner, MultiPassThoughtPlannerConfig
from .planner_evaluation import PlannerEvaluator, PlannerEvaluationConfig, _safe_jsonable
from .planner_failure_classifier import PlannerFailureClassifier, PlannerFailureClassifierConfig
from .planner_remediation_guidance import PlannerRemediationGuidance, PlannerRemediationGuidanceConfig
from .planner_quality_hardening import PlannerQualityHardener, PlannerQualityHardeningConfig


class ControllerPlannerIntegrationError(ValueError):
    """Raised when controller/planner integration violates safety contracts."""


@dataclass(frozen=True)
class ControllerPlannerIntegrationConfig:
    """Explicit opt-in controller/planner integration config."""

    enabled: bool = False
    run_planner: bool = False
    run_evaluation: bool = False
    run_failure_classifier: bool = False
    run_remediation_guidance: bool = False
    run_quality_hardening: bool = False
    planner_config: Optional[MultiPassThoughtPlannerConfig] = None
    evaluation_config: Optional[PlannerEvaluationConfig] = None
    classifier_config: Optional[PlannerFailureClassifierConfig] = None
    remediation_config: Optional[PlannerRemediationGuidanceConfig] = None
    hardening_config: Optional[PlannerQualityHardeningConfig] = None
    require_write_permission_false: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise ControllerPlannerIntegrationError("no_mutation_by_default must remain true")
        if not self.require_write_permission_false:
            raise ControllerPlannerIntegrationError("write permission must remain false for REASON-3C integration")
        if not self.enabled and any([self.run_planner, self.run_evaluation, self.run_failure_classifier, self.run_remediation_guidance, self.run_quality_hardening]):
            raise ControllerPlannerIntegrationError("sub-routes cannot run when integration is disabled")

    @classmethod
    def disabled(cls) -> "ControllerPlannerIntegrationConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ControllerPlannerIntegrationConfig":
        return cls(
            enabled=True,
            run_planner=True,
            run_evaluation=True,
            run_failure_classifier=True,
            run_remediation_guidance=True,
            run_quality_hardening=True,
            planner_config=MultiPassThoughtPlannerConfig.enabled_default(),
            evaluation_config=PlannerEvaluationConfig.enabled_default(),
            classifier_config=PlannerFailureClassifierConfig.enabled_default(),
            remediation_config=PlannerRemediationGuidanceConfig.enabled_default(),
            hardening_config=PlannerQualityHardeningConfig.enabled_default(),
        )


@dataclass
class ControllerPlannerIntegrationReport:
    """JSON-safe controller/planner integration report."""

    enabled: bool
    planner_report: Dict[str, Any] = field(default_factory=dict)
    evaluation_report: Dict[str, Any] = field(default_factory=dict)
    failure_records: list = field(default_factory=list)
    remediation_report: Dict[str, Any] = field(default_factory=dict)
    hardening_report: Dict[str, Any] = field(default_factory=dict)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"controller_planner_integration_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "planner_report": _safe_jsonable(self.planner_report),
            "evaluation_report": _safe_jsonable(self.evaluation_report),
            "failure_records": _safe_jsonable(self.failure_records),
            "remediation_report": _safe_jsonable(self.remediation_report),
            "hardening_report": _safe_jsonable(self.hardening_report),
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
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ControllerPlannerIntegration:
    """Runs explicit opt-in planner/evaluation/classification/hardening pipeline."""

    def __init__(self, config: Optional[ControllerPlannerIntegrationConfig] = None):
        self.config = config or ControllerPlannerIntegrationConfig.disabled()
        self.config.validate()

    def run(
        self,
        query: torch.Tensor,
        *,
        content: str = "",
        evidence_report: Optional[Any] = None,
        write_permission: bool = False,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> ControllerPlannerIntegrationReport:
        if write_permission:
            raise ControllerPlannerIntegrationError("write_permission=True is forbidden for REASON-3C planner integration")
        if not isinstance(query, torch.Tensor):
            raise ControllerPlannerIntegrationError("query must be a torch.Tensor")
        before = query.clone()
        if not self.config.enabled:
            return ControllerPlannerIntegrationReport(
                enabled=False,
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        planner_report_obj = None
        planner_payload: Dict[str, Any] = {}
        if self.config.run_planner:
            planner = MultiPassThoughtPlanner(self.config.planner_config or MultiPassThoughtPlannerConfig.enabled_default())
            planner_report_obj = planner.plan(query, content=content, evidence_report=evidence_report)
            planner_payload = planner_report_obj.to_dict()

        evaluation_obj = None
        evaluation_payload: Dict[str, Any] = {}
        if self.config.run_evaluation:
            evaluator = PlannerEvaluator(self.config.evaluation_config or PlannerEvaluationConfig.enabled_default())
            evaluation_obj = evaluator.evaluate(planner_report_obj or planner_payload, lineage=lineage)
            evaluation_payload = evaluation_obj.to_dict()

        failure_records = []
        if self.config.run_failure_classifier:
            classifier = PlannerFailureClassifier(self.config.classifier_config or PlannerFailureClassifierConfig.enabled_default())
            failure_records = classifier.classify(evaluation_obj or evaluation_payload, lineage=lineage)

        remediation_obj = None
        remediation_payload: Dict[str, Any] = {}
        if self.config.run_remediation_guidance:
            remediation = PlannerRemediationGuidance(self.config.remediation_config or PlannerRemediationGuidanceConfig.enabled_default())
            remediation_obj = remediation.recommend(failure_records, lineage=lineage)
            remediation_payload = remediation_obj.to_dict()

        hardening_payload: Dict[str, Any] = {}
        if self.config.run_quality_hardening:
            hardener = PlannerQualityHardener(self.config.hardening_config or PlannerQualityHardeningConfig.enabled_default())
            hardening_payload = hardener.harden(
                evaluation_report=evaluation_obj or evaluation_payload,
                failure_records=failure_records,
                remediation_report=remediation_obj or remediation_payload,
                lineage=lineage,
            ).to_dict()

        if not torch.equal(query, before):
            raise ControllerPlannerIntegrationError("query tensor was mutated")

        report = ControllerPlannerIntegrationReport(
            enabled=True,
            planner_report=planner_payload,
            evaluation_report=evaluation_payload,
            failure_records=[item.to_dict() if hasattr(item, "to_dict") else item for item in failure_records],
            remediation_report=remediation_payload,
            hardening_report=hardening_payload,
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "explicit_opt_in_required": True,
            "write_permission_allowed": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
            "automatic_patch_application": False,
            "real_ablation_execution": False,
        }


def controller_planner_integration_contract() -> Dict[str, Any]:
    return {
        "module": "controller_planner_integration",
        "stage": "REASON-3C",
        "default_enabled": False,
        "explicit_opt_in_required": True,
        "write_permission_allowed": False,
        "permanent_memory_store_mutation": False,
        "json_safe_report": True,
    }
