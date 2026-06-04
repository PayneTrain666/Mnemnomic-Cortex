from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_controller_api import reasoning_controller_api_contract
from .reasoning_release_audit import run_reasoning_release_audit, ReasoningReleaseAuditConfig
from .reasoning_regression_matrix import build_reasoning_regression_matrix
from .planner_quality_hardening import planner_quality_hardening_contract
from .controller_planner_integration import controller_planner_integration_contract
from .strategy_graph_persistence_readiness import strategy_graph_persistence_readiness_contract


class ReasoningReleaseCandidateError(ValueError):
    """Raised when release-candidate checks fail unsafe configuration."""


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


def _enabled_release_audit_config() -> ReasoningReleaseAuditConfig:
    """Construct a release audit config without assuming helper constructors exist."""
    try:
        return ReasoningReleaseAuditConfig(enabled=True)
    except TypeError:
        return ReasoningReleaseAuditConfig()


@dataclass(frozen=True)
class ReasoningReleaseCandidateConfig:
    """Release-candidate config for the current reasoning-controller line.

    This stage is a release-readiness layer only. It does not enable
    persistence, commit gates, memory-store mutation, model mutation, or
    optimizer mutation.
    """

    enabled: bool = False
    candidate_name: str = "mnemonic-reasoning-rc"
    require_api_freeze: bool = True
    require_regression_closure: bool = True
    require_release_audit: bool = True
    require_disabled_defaults: bool = True
    require_no_mutation: bool = True
    require_no_fake_production_claim: bool = True
    require_json_safe_outputs: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.candidate_name:
            raise ReasoningReleaseCandidateError("candidate_name is required")
        if not self.no_mutation_by_default:
            raise ReasoningReleaseCandidateError("no_mutation_by_default must remain true")
        if not self.require_no_mutation:
            raise ReasoningReleaseCandidateError("release candidate must require no-mutation checks")
        if not self.require_no_fake_production_claim:
            raise ReasoningReleaseCandidateError("release candidate must forbid fake production-complete claims")

    @classmethod
    def disabled(cls) -> "ReasoningReleaseCandidateConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningReleaseCandidateConfig":
        return cls(enabled=True)


@dataclass
class ReasoningReleaseCandidateReport:
    """JSON-safe release-candidate report."""

    enabled: bool
    candidate_name: str
    readiness_level: str
    pass_gate: bool
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    regression_summary: Dict[str, Any] = field(default_factory=dict)
    api_summary: Dict[str, Any] = field(default_factory=dict)
    blocking_items: List[str] = field(default_factory=list)
    deferred_items: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"reasoning_release_candidate_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "candidate_name": self.candidate_name,
            "readiness_level": self.readiness_level,
            "pass_gate": bool(self.pass_gate),
            "audit_summary": _safe_jsonable(self.audit_summary),
            "regression_summary": _safe_jsonable(self.regression_summary),
            "api_summary": _safe_jsonable(self.api_summary),
            "blocking_items": list(self.blocking_items),
            "deferred_items": list(self.deferred_items),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
                "policy_lane_integration": True,
                "confidence_hook": True,
                "disagreement_hook": True,
                "conflict_hook": True,
                "quarantine_hook": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningReleaseCandidate:
    """Builds a release-candidate readiness report without mutating runtime state."""

    def __init__(self, config: Optional[ReasoningReleaseCandidateConfig] = None):
        self.config = config or ReasoningReleaseCandidateConfig.disabled()
        self.config.validate()

    def evaluate(self, *, lineage: Optional[Dict[str, Any]] = None) -> ReasoningReleaseCandidateReport:
        if not self.config.enabled:
            return ReasoningReleaseCandidateReport(
                enabled=False,
                candidate_name=self.config.candidate_name,
                readiness_level="disabled",
                pass_gate=False,
                blocking_items=["release_candidate_disabled"],
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        audit = run_reasoning_release_audit(_enabled_release_audit_config()).to_dict()
        regression = build_reasoning_regression_matrix().to_dict()
        api_contract = reasoning_controller_api_contract()

        blocking: List[str] = []
        audit_readiness = audit.get("readiness_level") or audit.get("readiness") or "unknown"
        if self.config.require_release_audit and str(audit_readiness).lower() in {"blocked", "fail", "failed"}:
            blocking.append("release_audit_not_ready")
        if self.config.require_regression_closure and len(regression.get("rows", [])) < 10:
            blocking.append("regression_matrix_incomplete")
        if self.config.require_disabled_defaults and api_contract.get("default_enabled") is not False:
            blocking.append("api_default_not_disabled")
        if api_contract.get("write_permission_public_api") is not False:
            blocking.append("public_api_allows_write_permission")
        if controller_planner_integration_contract().get("write_permission_allowed") is not False:
            blocking.append("controller_planner_integration_allows_writes")
        if strategy_graph_persistence_readiness_contract().get("automatic_persistence") is not False:
            blocking.append("strategy_graph_automatic_persistence_enabled")
        if planner_quality_hardening_contract().get("automatic_patch_application") is not False:
            blocking.append("planner_quality_auto_patch_enabled")

        pass_gate = not blocking
        readiness = "release_candidate" if pass_gate else "blocked"
        report = ReasoningReleaseCandidateReport(
            enabled=True,
            candidate_name=self.config.candidate_name,
            readiness_level=readiness,
            pass_gate=pass_gate,
            audit_summary={
                "readiness_level": audit_readiness,
                "module_availability": audit.get("module_availability", {}),
                "disabled_default_verification": audit.get("disabled_default_verification"),
                "no_mutation_verification": audit.get("no_mutation_verification"),
            },
            regression_summary={
                "row_count": len(regression.get("rows", [])),
                "stage_count": len({row.get("stage") for row in regression.get("rows", []) if isinstance(row, dict)}),
                "summary": regression.get("summary", {}),
                "json_safe": True,
            },
            api_summary=api_contract,
            blocking_items=blocking,
            deferred_items=[
                "production persistence adapter implementation",
                "permanent consolidation commit API behind explicit gate",
                "external store integration",
                "long-run benchmark campaign",
            ],
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        if self.config.require_json_safe_outputs:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "release_readiness_only": True,
            "no_fake_production_complete_claim": True,
            "permanent_memory_store_mutation": False,
            "automatic_persistence": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
            "automatic_policy_activation": False,
            "automatic_component_activation": False,
        }


def reasoning_release_candidate_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_release_candidate",
        "stage": "REASON-3D",
        "default_enabled": False,
        "release_readiness_only": True,
        "no_fake_production_complete_claim": True,
        "permanent_memory_store_mutation": False,
        "json_safe_report": True,
    }
