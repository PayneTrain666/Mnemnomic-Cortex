from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from .evidence_reasoning_pass import EvidenceReasoningReport
from .reasoning_orchestration_trace import _safe_jsonable


class CounterfactualProbeError(ValueError):
    """Raised when counterfactual probing receives unsafe input."""


@dataclass(frozen=True)
class CounterfactualProbeConfig:
    """Bounded counterfactual probe config.

    REASON-2C probes are metadata-level ablations over evidence units. They do
    not rerun model computation, mutate tensors, mutate memory stores, or apply
    patches.
    """

    enabled: bool = False
    max_probes: int = 8
    disagreement_threshold: float = 0.45
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_probes <= 0 or self.max_probes > 128:
            raise CounterfactualProbeError("max_probes must be in [1,128]")
        if not (0.0 <= self.disagreement_threshold <= 1.0):
            raise CounterfactualProbeError("disagreement_threshold must be in [0,1]")

    @classmethod
    def disabled(cls) -> "CounterfactualProbeConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "CounterfactualProbeConfig":
        return cls(enabled=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_probes": self.max_probes,
            "disagreement_threshold": self.disagreement_threshold,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class CounterfactualProbeItem:
    """One bounded counterfactual evidence ablation result."""

    removed_unit_id: str
    estimated_confidence_delta: float
    estimated_disagreement_delta: float
    probe_id: str = field(default_factory=lambda: f"cf_probe_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "removed_unit_id": self.removed_unit_id,
            "estimated_confidence_delta": float(self.estimated_confidence_delta),
            "estimated_disagreement_delta": float(self.estimated_disagreement_delta),
            "metadata": _safe_jsonable(self.metadata),
        }


@dataclass
class CounterfactualProbeReport:
    """JSON-safe counterfactual probe report."""

    enabled: bool
    probes: List[CounterfactualProbeItem]
    report_id: str = field(default_factory=lambda: f"counterfactual_report_{uuid.uuid4().hex[:16]}")
    max_disagreement_delta: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probes:
            self.max_disagreement_delta = float(max(abs(p.estimated_disagreement_delta) for p in self.probes))
        else:
            self.max_disagreement_delta = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "enabled": self.enabled,
            "probes": [probe.to_dict() for probe in self.probes],
            "max_disagreement_delta": float(self.max_disagreement_delta),
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "counterfactual_probe": True,
                "trace_governance": True,
                "audit_metadata": True,
                "no_real_ablation_execution": True,
            },
            "safety": {
                "bounded_probes": True,
                "non_mutating": True,
                "proposal_only": True,
            },
        }


class CounterfactualReasoningProbe:
    """Metadata-only bounded counterfactual probe pass."""

    def __init__(self, config: Optional[CounterfactualProbeConfig] = None):
        self.config = config or CounterfactualProbeConfig.disabled()
        self.config.validate()

    def run(
        self,
        evidence_report: EvidenceReasoningReport,
        *,
        base_confidence: float = 0.75,
        base_disagreement: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualProbeReport:
        if not isinstance(evidence_report, EvidenceReasoningReport):
            raise CounterfactualProbeError("evidence_report must be EvidenceReasoningReport")

        if not self.config.enabled:
            return CounterfactualProbeReport(
                enabled=False,
                probes=[],
                metadata={"reason": "counterfactual probing disabled", **(metadata or {})},
            )

        probes: List[CounterfactualProbeItem] = []
        units = evidence_report.evidence_units[: self.config.max_probes]
        for index, unit in enumerate(units):
            support = float(max(0.0, min(1.0, unit.support)))
            confidence_delta = -0.2 * support
            disagreement_delta = 0.2 * support
            probes.append(
                CounterfactualProbeItem(
                    removed_unit_id=unit.unit_id,
                    estimated_confidence_delta=confidence_delta,
                    estimated_disagreement_delta=disagreement_delta,
                    metadata={
                        "index": index,
                        "base_confidence": float(base_confidence),
                        "base_disagreement": float(base_disagreement),
                        "support": support,
                    },
                )
            )

        return CounterfactualProbeReport(
            enabled=True,
            probes=probes,
            metadata={"evidence_report_id": evidence_report.report_id, **(metadata or {})},
        )


def counterfactual_reasoning_contract() -> Dict[str, Any]:
    return {
        "module": "counterfactual_reasoning_probe",
        "stage": "REASON-2C",
        "default_enabled": False,
        "bounded_probes": True,
        "non_mutating": True,
        "no_real_ablation_execution": True,
    }
