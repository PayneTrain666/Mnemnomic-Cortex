"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: conflict aware consolidation.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid

from .counterfactual_reasoning_probe import CounterfactualProbeReport
from .evidence_reasoning_pass import EvidenceReasoningReport
from .reasoning_orchestration_trace import _safe_jsonable


class ConflictAwareConsolidationError(ValueError):
    """Raised when conflict-aware consolidation receives unsafe input."""


@dataclass(frozen=True)
class ConflictAwareConsolidationConfig:
    """Conflict-aware consolidation evaluator.

    This module produces consolidation safety metadata. It does not commit to
    memory stores. Quarantine means "route to quarantine metadata", not a
    destructive action.
    """

    enabled: bool = False
    disagreement_quarantine_threshold: float = 0.65
    counterfactual_delta_threshold: float = 0.20
    min_evidence_support_for_commit_ready: float = 0.35
    no_commit_by_default: bool = True

    def validate(self) -> None:
        for name, value in [
            ("disagreement_quarantine_threshold", self.disagreement_quarantine_threshold),
            ("counterfactual_delta_threshold", self.counterfactual_delta_threshold),
            ("min_evidence_support_for_commit_ready", self.min_evidence_support_for_commit_ready),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ConflictAwareConsolidationError(f"{name} must be in [0,1]")

    @classmethod
    def disabled(cls) -> "ConflictAwareConsolidationConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ConflictAwareConsolidationConfig":
        return cls(enabled=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "disagreement_quarantine_threshold": self.disagreement_quarantine_threshold,
            "counterfactual_delta_threshold": self.counterfactual_delta_threshold,
            "min_evidence_support_for_commit_ready": self.min_evidence_support_for_commit_ready,
            "no_commit_by_default": self.no_commit_by_default,
        }


@dataclass
class ConflictAwareConsolidationReport:
    """JSON-safe consolidation safety report."""

    enabled: bool
    conflict_detected: bool
    quarantine_recommended: bool
    adjusted_confidence: float
    adjusted_disagreement: float
    reason: str
    report_id: str = field(default_factory=lambda: f"conflict_report_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "enabled": self.enabled,
            "conflict_detected": self.conflict_detected,
            "quarantine_recommended": self.quarantine_recommended,
            "adjusted_confidence": float(self.adjusted_confidence),
            "adjusted_disagreement": float(self.adjusted_disagreement),
            "reason": self.reason,
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "conflict_hook": True,
                "quarantine_hook": True,
                "trace_governance": True,
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
            },
            "safety": {
                "non_mutating": True,
                "no_commit_by_default": True,
                "quarantine_is_metadata_only": True,
            },
        }


class ConflictAwareConsolidationEvaluator:
    """Evaluates conflict/quarantine metadata for consolidation proposals."""

    def __init__(self, config: Optional[ConflictAwareConsolidationConfig] = None):
        self.config = config or ConflictAwareConsolidationConfig.disabled()
        self.config.validate()

    def evaluate(
        self,
        *,
        evidence_report: Optional[EvidenceReasoningReport],
        counterfactual_report: Optional[CounterfactualProbeReport],
        confidence: float,
        disagreement: float,
        base_conflict: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConflictAwareConsolidationReport:
        confidence = float(max(0.0, min(1.0, confidence)))
        disagreement = float(max(0.0, min(1.0, disagreement)))

        if not self.config.enabled:
            return ConflictAwareConsolidationReport(
                enabled=False,
                conflict_detected=bool(base_conflict),
                quarantine_recommended=bool(base_conflict),
                adjusted_confidence=confidence,
                adjusted_disagreement=disagreement,
                reason="conflict-aware consolidation disabled",
                metadata=metadata or {},
            )

        support = evidence_report.aggregate_support if evidence_report is not None else 0.0
        max_cf_delta = counterfactual_report.max_disagreement_delta if counterfactual_report is not None else 0.0

        adjusted_disagreement = min(1.0, disagreement + max_cf_delta * 0.5 + max(0.0, 0.3 - support) * 0.25)
        adjusted_confidence = max(0.0, min(1.0, confidence - max_cf_delta * 0.25))

        conflict_detected = bool(base_conflict or adjusted_disagreement >= self.config.disagreement_quarantine_threshold)
        quarantine_recommended = bool(
            conflict_detected
            or max_cf_delta >= self.config.counterfactual_delta_threshold
            or support < self.config.min_evidence_support_for_commit_ready
        )

        if quarantine_recommended:
            reason = "quarantine recommended by disagreement/counterfactual/evidence thresholds"
        else:
            reason = "no conflict threshold exceeded; remain shadow-only unless later explicit commit gate approves"

        return ConflictAwareConsolidationReport(
            enabled=True,
            conflict_detected=conflict_detected,
            quarantine_recommended=quarantine_recommended,
            adjusted_confidence=adjusted_confidence,
            adjusted_disagreement=adjusted_disagreement,
            reason=reason,
            metadata={
                "evidence_support": support,
                "max_counterfactual_delta": max_cf_delta,
                **(metadata or {}),
            },
        )


def conflict_aware_consolidation_contract() -> Dict[str, Any]:
    return {
        "module": "conflict_aware_consolidation",
        "stage": "REASON-2C",
        "default_enabled": False,
        "non_mutating": True,
        "conflict_quarantine_hooks": True,
        "no_commit_by_default": True,
    }
