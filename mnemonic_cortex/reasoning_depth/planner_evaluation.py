from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid


class PlannerEvaluationError(ValueError):
    """Raised when planner evaluation violates REASON-3B safety contracts."""


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
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class PlannerEvaluationConfig:
    """Planner evaluation config. Disabled by default and non-mutating."""

    enabled: bool = False
    max_plan_passes: int = 16
    max_route_candidates: int = 64
    confidence_floor: float = 0.35
    disagreement_ceiling: float = 0.75
    evidence_support_floor: float = 0.20
    conflict_penalty: float = 0.20
    unsupported_penalty: float = 0.15
    finite_checks: bool = True
    require_json_safe_outputs: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_plan_passes <= 0 or self.max_plan_passes > 1024:
            raise PlannerEvaluationError("max_plan_passes must be in [1,1024]")
        if self.max_route_candidates <= 0 or self.max_route_candidates > 4096:
            raise PlannerEvaluationError("max_route_candidates must be in [1,4096]")
        for name in ("confidence_floor", "disagreement_ceiling", "evidence_support_floor", "conflict_penalty", "unsupported_penalty"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise PlannerEvaluationError(f"{name} must be in [0,1]")
        if not self.no_mutation_by_default:
            raise PlannerEvaluationError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PlannerEvaluationConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PlannerEvaluationConfig":
        return cls(enabled=True)


@dataclass(frozen=True)
class PlannerEvaluationScore:
    """JSON-safe planner quality score."""

    pass_count: int
    route_count: int
    mean_confidence: float
    max_disagreement: float
    mean_evidence_support: float
    unsupported_route_count: int
    conflict_route_count: int
    boundedness_ok: bool
    no_mutation_ok: bool
    overall_score: float
    score_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.score_id is None:
            object.__setattr__(self, "score_id", _stable_id("planner_score", self.to_dict(include_id=False)))

    def to_dict(self, include_id: bool = True) -> Dict[str, Any]:
        payload = {
            "pass_count": int(self.pass_count),
            "route_count": int(self.route_count),
            "mean_confidence": float(self.mean_confidence),
            "max_disagreement": float(self.max_disagreement),
            "mean_evidence_support": float(self.mean_evidence_support),
            "unsupported_route_count": int(self.unsupported_route_count),
            "conflict_route_count": int(self.conflict_route_count),
            "boundedness_ok": bool(self.boundedness_ok),
            "no_mutation_ok": bool(self.no_mutation_ok),
            "overall_score": float(self.overall_score),
        }
        if include_id:
            payload["score_id"] = self.score_id
        return payload


@dataclass
class PlannerEvaluationReport:
    """JSON-safe evaluation report."""

    enabled: bool
    source_plan_id: Optional[str]
    score: PlannerEvaluationScore
    failure_signals: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"planner_eval_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "source_plan_id": self.source_plan_id,
            "score": self.score.to_dict(),
            "failure_signals": list(self.failure_signals),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "confidence_hook": True,
                "disagreement_hook": True,
                "conflict_hook": True,
                "quarantine_hook": True,
                "audit_metadata": True,
                "policy_lane_integration": True,
                "write_permission_required_for_commit": True,
            },
        }
        json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)


class PlannerEvaluator:
    """Evaluates ThoughtPlanReport payloads without mutation or remediation."""

    def __init__(self, config: Optional[PlannerEvaluationConfig] = None):
        self.config = config or PlannerEvaluationConfig.disabled()
        self.config.validate()

    def evaluate(self, plan: Optional[Any], *, lineage: Optional[Dict[str, Any]] = None) -> PlannerEvaluationReport:
        if not self.config.enabled:
            return PlannerEvaluationReport(
                enabled=False,
                source_plan_id=None,
                score=self._zero_score(no_mutation_ok=True),
                failure_signals=["evaluation_disabled"],
                safety_flags=self._safety_flags(no_mutation_ok=True),
                lineage=lineage or {},
            )

        payload = self._as_payload(plan)
        if not payload or payload.get("enabled") is False:
            return PlannerEvaluationReport(
                enabled=True,
                source_plan_id=payload.get("report_id") if isinstance(payload, dict) else None,
                score=self._zero_score(no_mutation_ok=True),
                failure_signals=["disabled_or_missing_plan"],
                safety_flags=self._safety_flags(no_mutation_ok=True),
                lineage=lineage or {},
            )

        passes = payload.get("passes", [])
        if not isinstance(passes, list):
            passes = []
        route_candidates = self._extract_route_candidates(payload)
        pass_count = len(passes)
        route_count = len(route_candidates)

        confidences = [float(item.get("confidence", 0.0)) for item in passes if isinstance(item, dict)]
        disagreements = [float(item.get("disagreement", 0.0)) for item in passes if isinstance(item, dict)]
        supports = [float(item.get("evidence_support", 0.0)) for item in passes if isinstance(item, dict)]
        unsupported_count = sum(1 for item in route_candidates if isinstance(item, dict) and bool(item.get("unsupported", False)))
        conflict_count = sum(1 for item in route_candidates if isinstance(item, dict) and bool(item.get("conflict_prone", False)))

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        max_dis = max(disagreements) if disagreements else 0.0
        mean_support = sum(supports) / len(supports) if supports else 0.0
        boundedness_ok = pass_count <= self.config.max_plan_passes and route_count <= self.config.max_route_candidates
        safety_payload = payload.get("safety", {}) if isinstance(payload.get("safety", {}), dict) else {}
        no_mutation_ok = not bool(safety_payload.get("permanent_memory_store_mutation", False))

        score_raw = mean_conf
        score_raw += mean_support * 0.20
        score_raw -= max(0.0, max_dis - self.config.disagreement_ceiling) * 0.30
        score_raw -= conflict_count * self.config.conflict_penalty
        score_raw -= unsupported_count * self.config.unsupported_penalty
        if not boundedness_ok:
            score_raw -= 0.25
        if not no_mutation_ok:
            score_raw -= 1.0
        overall = max(0.0, min(1.0, score_raw))

        failure_signals: List[str] = []
        if pass_count == 0:
            failure_signals.append("empty_plan")
        if pass_count > self.config.max_plan_passes:
            failure_signals.append("excessive_pass_count")
        if route_count > self.config.max_route_candidates:
            failure_signals.append("excessive_route_count")
        if mean_conf < self.config.confidence_floor:
            failure_signals.append("low_confidence")
        if max_dis > self.config.disagreement_ceiling:
            failure_signals.append("high_disagreement")
        if mean_support < self.config.evidence_support_floor:
            failure_signals.append("low_evidence_support")
        if conflict_count:
            failure_signals.append("conflict_prone_route")
        if unsupported_count:
            failure_signals.append("unsupported_route")
        if not no_mutation_ok:
            failure_signals.append("unsafe_mutation_signal")

        score = PlannerEvaluationScore(
            pass_count=pass_count,
            route_count=route_count,
            mean_confidence=mean_conf,
            max_disagreement=max_dis,
            mean_evidence_support=mean_support,
            unsupported_route_count=unsupported_count,
            conflict_route_count=conflict_count,
            boundedness_ok=boundedness_ok,
            no_mutation_ok=no_mutation_ok,
            overall_score=overall,
        )
        report = PlannerEvaluationReport(
            enabled=True,
            source_plan_id=payload.get("report_id"),
            score=score,
            failure_signals=failure_signals,
            safety_flags=self._safety_flags(no_mutation_ok=no_mutation_ok, boundedness_ok=boundedness_ok),
            lineage=lineage or {},
        )
        if self.config.require_json_safe_outputs:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    def _zero_score(self, *, no_mutation_ok: bool) -> PlannerEvaluationScore:
        return PlannerEvaluationScore(
            pass_count=0,
            route_count=0,
            mean_confidence=0.0,
            max_disagreement=0.0,
            mean_evidence_support=0.0,
            unsupported_route_count=0,
            conflict_route_count=0,
            boundedness_ok=True,
            no_mutation_ok=no_mutation_ok,
            overall_score=0.0,
        )

    @staticmethod
    def _as_payload(plan: Optional[Any]) -> Dict[str, Any]:
        if plan is None:
            return {}
        if hasattr(plan, "to_dict"):
            payload = plan.to_dict()
        elif isinstance(plan, dict):
            payload = plan
        else:
            raise PlannerEvaluationError("plan must be ThoughtPlanReport-like or dict")
        json.dumps(_safe_jsonable(payload), sort_keys=True)
        return _safe_jsonable(payload)

    @staticmethod
    def _extract_route_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        expansion = payload.get("route_expansion")
        if isinstance(expansion, dict):
            candidates = expansion.get("candidates", [])
            if isinstance(candidates, list):
                return candidates
        return []

    @staticmethod
    def _safety_flags(*, no_mutation_ok: bool, boundedness_ok: bool = True) -> Dict[str, Any]:
        return {
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "automatic_remediation": False,
            "real_ablation_execution": False,
            "no_mutation_ok": bool(no_mutation_ok),
            "boundedness_ok": bool(boundedness_ok),
        }


def planner_evaluation_contract() -> Dict[str, Any]:
    return {
        "module": "planner_evaluation",
        "stage": "REASON-3B",
        "default_enabled": False,
        "recommendation_only": True,
        "permanent_memory_store_mutation": False,
        "automatic_remediation": False,
        "json_safe_report": True,
        "paamax_metadata": True,
    }
