from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json

from .planner_evaluation import PlannerEvaluationReport, _safe_jsonable


class PlannerFailureClassifierError(ValueError):
    """Raised when planner failure classification cannot proceed safely."""


class PlannerFailureFamily(str, Enum):
    DISABLED_OR_MISSING_PLAN = "disabled_or_missing_plan"
    EMPTY_PLAN = "empty_plan"
    EXCESSIVE_PASS_COUNT = "excessive_pass_count"
    EXCESSIVE_ROUTE_COUNT = "excessive_route_count"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_DISAGREEMENT = "high_disagreement"
    LOW_EVIDENCE_SUPPORT = "low_evidence_support"
    CONFLICT_PRONE_ROUTE = "conflict_prone_route"
    UNSUPPORTED_ROUTE = "unsupported_route"
    MALFORMED_TRACE = "malformed_trace"
    NON_JSON_SAFE_PAYLOAD = "non_json_safe_payload"
    UNSAFE_MUTATION_SIGNAL = "unsafe_mutation_signal"
    UNKNOWN = "unknown"


class PlannerFailureSeverity(str, Enum):
    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


def _stable_failure_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(_safe_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return f"planner_failure_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class PlannerFailureRecord:
    """A bounded, deterministic, JSON-safe planner failure record."""

    family: PlannerFailureFamily
    severity: PlannerFailureSeverity
    reason: str
    affected_pass_ids: List[str] = field(default_factory=list)
    affected_route_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    remediation_hint: str = ""
    failure_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.failure_id is None:
            object.__setattr__(
                self,
                "failure_id",
                _stable_failure_id(
                    {
                        "family": self.family.value,
                        "severity": self.severity.value,
                        "reason": self.reason,
                        "affected_pass_ids": list(self.affected_pass_ids),
                        "affected_route_ids": list(self.affected_route_ids),
                        "lineage": self.lineage,
                    }
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "failure_id": self.failure_id,
                "family": self.family.value,
                "severity": self.severity.value,
                "reason": self.reason,
                "affected_pass_ids": list(self.affected_pass_ids),
                "affected_route_ids": list(self.affected_route_ids),
                "evidence": self.evidence,
                "lineage": self.lineage,
                "remediation_hint": self.remediation_hint,
            }
        )


@dataclass(frozen=True)
class PlannerFailureClassifierConfig:
    """Failure classifier config. Disabled by default and non-mutating."""

    enabled: bool = False
    max_failures: int = 32
    confidence_floor: float = 0.35
    disagreement_ceiling: float = 0.75
    evidence_support_floor: float = 0.20
    max_passes: int = 16
    max_routes: int = 64
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_failures <= 0 or self.max_failures > 1024:
            raise PlannerFailureClassifierError("max_failures must be in [1,1024]")
        if self.max_passes <= 0 or self.max_routes <= 0:
            raise PlannerFailureClassifierError("max_passes and max_routes must be positive")
        for name in ("confidence_floor", "disagreement_ceiling", "evidence_support_floor"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise PlannerFailureClassifierError(f"{name} must be in [0,1]")
        if not self.no_mutation_by_default:
            raise PlannerFailureClassifierError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PlannerFailureClassifierConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PlannerFailureClassifierConfig":
        return cls(enabled=True)


class PlannerFailureClassifier:
    """Classifies planner failure signals into bounded failure records."""

    def __init__(self, config: Optional[PlannerFailureClassifierConfig] = None):
        self.config = config or PlannerFailureClassifierConfig.disabled()
        self.config.validate()

    def classify(self, evaluation_report: Any, *, lineage: Optional[Dict[str, Any]] = None) -> List[PlannerFailureRecord]:
        if not self.config.enabled:
            return []
        try:
            payload = self._as_payload(evaluation_report)
        except Exception as exc:
            return [
                PlannerFailureRecord(
                    family=PlannerFailureFamily.NON_JSON_SAFE_PAYLOAD,
                    severity=PlannerFailureSeverity.BLOCKER,
                    reason="evaluation report could not be converted to JSON-safe payload",
                    evidence={"error": str(exc)},
                    lineage=lineage or {},
                    remediation_hint="Ensure PlannerEvaluationReport.to_dict() returns JSON-safe data.",
                )
            ][: self.config.max_failures]

        records: List[PlannerFailureRecord] = []
        score = payload.get("score", {}) if isinstance(payload.get("score", {}), dict) else {}
        signals = payload.get("failure_signals", [])
        if not isinstance(signals, list):
            records.append(
                self._record(
                    PlannerFailureFamily.MALFORMED_TRACE,
                    PlannerFailureSeverity.HIGH,
                    "failure_signals is not a list",
                    evidence={"failure_signals_type": type(signals).__name__},
                    lineage=lineage,
                )
            )
            signals = []

        mapping = {
            "disabled_or_missing_plan": (PlannerFailureFamily.DISABLED_OR_MISSING_PLAN, PlannerFailureSeverity.INFO, "Planner is disabled or missing."),
            "empty_plan": (PlannerFailureFamily.EMPTY_PLAN, PlannerFailureSeverity.HIGH, "Planner produced no passes."),
            "excessive_pass_count": (PlannerFailureFamily.EXCESSIVE_PASS_COUNT, PlannerFailureSeverity.HIGH, "Planner exceeded pass budget."),
            "excessive_route_count": (PlannerFailureFamily.EXCESSIVE_ROUTE_COUNT, PlannerFailureSeverity.HIGH, "Planner exceeded route budget."),
            "low_confidence": (PlannerFailureFamily.LOW_CONFIDENCE, PlannerFailureSeverity.MEDIUM, "Planner confidence is below floor."),
            "high_disagreement": (PlannerFailureFamily.HIGH_DISAGREEMENT, PlannerFailureSeverity.MEDIUM, "Planner disagreement exceeds ceiling."),
            "low_evidence_support": (PlannerFailureFamily.LOW_EVIDENCE_SUPPORT, PlannerFailureSeverity.MEDIUM, "Planner evidence support is below floor."),
            "conflict_prone_route": (PlannerFailureFamily.CONFLICT_PRONE_ROUTE, PlannerFailureSeverity.HIGH, "Planner selected or exposed conflict-prone route."),
            "unsupported_route": (PlannerFailureFamily.UNSUPPORTED_ROUTE, PlannerFailureSeverity.MEDIUM, "Planner exposed unsupported route."),
            "unsafe_mutation_signal": (PlannerFailureFamily.UNSAFE_MUTATION_SIGNAL, PlannerFailureSeverity.BLOCKER, "Planner payload indicates unsafe mutation."),
        }

        for signal in signals:
            family, severity, reason = mapping.get(str(signal), (PlannerFailureFamily.UNKNOWN, PlannerFailureSeverity.LOW, f"Unknown planner signal: {signal}"))
            records.append(self._record(family, severity, reason, evidence={"signal": signal, "score": score}, lineage=lineage))

        # Defensive independent checks in case signals were omitted.
        if int(score.get("pass_count", 0)) > self.config.max_passes:
            records.append(self._record(PlannerFailureFamily.EXCESSIVE_PASS_COUNT, PlannerFailureSeverity.HIGH, "pass_count exceeds max_passes", evidence=score, lineage=lineage))
        if int(score.get("route_count", 0)) > self.config.max_routes:
            records.append(self._record(PlannerFailureFamily.EXCESSIVE_ROUTE_COUNT, PlannerFailureSeverity.HIGH, "route_count exceeds max_routes", evidence=score, lineage=lineage))
        if float(score.get("mean_confidence", 1.0)) < self.config.confidence_floor:
            records.append(self._record(PlannerFailureFamily.LOW_CONFIDENCE, PlannerFailureSeverity.MEDIUM, "mean confidence below configured floor", evidence=score, lineage=lineage))
        if float(score.get("max_disagreement", 0.0)) > self.config.disagreement_ceiling:
            records.append(self._record(PlannerFailureFamily.HIGH_DISAGREEMENT, PlannerFailureSeverity.MEDIUM, "max disagreement above configured ceiling", evidence=score, lineage=lineage))
        if float(score.get("mean_evidence_support", 1.0)) < self.config.evidence_support_floor:
            records.append(self._record(PlannerFailureFamily.LOW_EVIDENCE_SUPPORT, PlannerFailureSeverity.MEDIUM, "mean evidence support below configured floor", evidence=score, lineage=lineage))
        if not bool(score.get("no_mutation_ok", True)):
            records.append(self._record(PlannerFailureFamily.UNSAFE_MUTATION_SIGNAL, PlannerFailureSeverity.BLOCKER, "no_mutation_ok is false", evidence=score, lineage=lineage))

        dedup: Dict[str, PlannerFailureRecord] = {}
        for record in records:
            dedup[record.failure_id] = record
        return list(dedup.values())[: self.config.max_failures]

    @staticmethod
    def _as_payload(report: Any) -> Dict[str, Any]:
        if hasattr(report, "to_dict"):
            payload = report.to_dict()
        elif isinstance(report, dict):
            payload = report
        else:
            raise PlannerFailureClassifierError("evaluation_report must be report-like or dict")
        json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)

    @staticmethod
    def _record(
        family: PlannerFailureFamily,
        severity: PlannerFailureSeverity,
        reason: str,
        *,
        evidence: Optional[Dict[str, Any]] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> PlannerFailureRecord:
        hint = {
            PlannerFailureFamily.EMPTY_PLAN: "Inspect planner graph construction and route candidate extraction.",
            PlannerFailureFamily.LOW_CONFIDENCE: "Increase evidence support, tighten route selection, or request more context.",
            PlannerFailureFamily.HIGH_DISAGREEMENT: "Route through conflict-aware evaluation before consolidation.",
            PlannerFailureFamily.LOW_EVIDENCE_SUPPORT: "Require stronger evidence units before accepting route.",
            PlannerFailureFamily.CONFLICT_PRONE_ROUTE: "Quarantine route and require counter-evidence review.",
            PlannerFailureFamily.UNSUPPORTED_ROUTE: "Demote unsupported route or add supporting evidence.",
            PlannerFailureFamily.UNSAFE_MUTATION_SIGNAL: "Fail closed and block any commit/write path.",
        }.get(family, "Review planner trace and route metadata.")
        return PlannerFailureRecord(
            family=family,
            severity=severity,
            reason=reason,
            evidence=evidence or {},
            lineage=lineage or {},
            remediation_hint=hint,
        )


def planner_failure_classifier_contract() -> Dict[str, Any]:
    return {
        "module": "planner_failure_classifier",
        "stage": "REASON-3B",
        "default_enabled": False,
        "bounded_failure_records": True,
        "deterministic_failure_ids": True,
        "automatic_patching": False,
        "permanent_memory_store_mutation": False,
        "json_safe_records": True,
    }
