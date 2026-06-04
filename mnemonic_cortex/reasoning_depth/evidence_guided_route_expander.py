from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_strategy_graph import _safe_jsonable, _stable_id


class EvidenceGuidedRouteExpanderError(ValueError):
    """Raised when evidence-guided route expansion is unsafe."""


@dataclass(frozen=True)
class EvidenceGuidedRouteExpanderConfig:
    """Evidence route expander config. Disabled by default."""

    enabled: bool = False
    max_evidence_units: int = 16
    max_route_candidates: int = 16
    confidence_floor: float = 0.25
    disagreement_ceiling: float = 0.75
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_evidence_units <= 0 or self.max_evidence_units > 512:
            raise EvidenceGuidedRouteExpanderError("max_evidence_units must be in [1,512]")
        if self.max_route_candidates <= 0 or self.max_route_candidates > 512:
            raise EvidenceGuidedRouteExpanderError("max_route_candidates must be in [1,512]")
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise EvidenceGuidedRouteExpanderError("confidence_floor must be in [0,1]")
        if not (0.0 <= self.disagreement_ceiling <= 1.0):
            raise EvidenceGuidedRouteExpanderError("disagreement_ceiling must be in [0,1]")
        if not self.no_mutation_by_default:
            raise EvidenceGuidedRouteExpanderError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "EvidenceGuidedRouteExpanderConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "EvidenceGuidedRouteExpanderConfig":
        return cls(enabled=True)


@dataclass
class EvidenceGuidedRouteCandidate:
    """Expanded route candidate with evidence support metadata."""

    route_id: str
    selected_depth_roles: List[str]
    confidence: float
    disagreement: float
    evidence_support: float
    unsupported: bool
    conflict_prone: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "route_id": self.route_id,
                "selected_depth_roles": list(self.selected_depth_roles),
                "confidence": float(self.confidence),
                "disagreement": float(self.disagreement),
                "evidence_support": float(self.evidence_support),
                "unsupported": bool(self.unsupported),
                "conflict_prone": bool(self.conflict_prone),
                "metadata": self.metadata,
            }
        )


@dataclass
class EvidenceGuidedRouteExpansionReport:
    """JSON-safe expansion report."""

    enabled: bool
    candidates: List[EvidenceGuidedRouteCandidate]
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"route_expansion_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "report_id": self.report_id,
                "enabled": self.enabled,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "metadata": self.metadata,
                "paamax_metadata": {
                    "trace_governance": True,
                    "confidence_hook": True,
                    "disagreement_hook": True,
                    "conflict_hook": True,
                    "policy_lane_integration": True,
                    "write_permission_required_for_commit": True,
                },
                "safety": {
                    "permanent_memory_store_mutation": False,
                    "bounded": True,
                    "real_ablation_execution": False,
                },
            }
        )


class EvidenceGuidedRouteExpander:
    """Scores and filters route candidates using evidence metadata only."""

    def __init__(self, config: Optional[EvidenceGuidedRouteExpanderConfig] = None):
        self.config = config or EvidenceGuidedRouteExpanderConfig.disabled()
        self.config.validate()

    def expand(self, route_candidates: List[Dict[str, Any]], evidence_report: Optional[Any] = None) -> EvidenceGuidedRouteExpansionReport:
        if not self.config.enabled:
            return EvidenceGuidedRouteExpansionReport(
                enabled=False,
                candidates=[],
                metadata={"reason": "expander disabled", "input_route_candidates": len(route_candidates)},
            )

        evidence_units = self._extract_evidence_units(evidence_report)[: self.config.max_evidence_units]
        evidence_count = len(evidence_units)
        evidence_support_base = min(1.0, evidence_count / max(1, self.config.max_evidence_units))

        expanded: List[EvidenceGuidedRouteCandidate] = []
        for raw in route_candidates[: self.config.max_route_candidates]:
            route_id = str(raw.get("route_id") or _stable_id("route", raw))
            raw_confidence = float(raw.get("confidence", 0.5))
            raw_disagreement = float(raw.get("disagreement", 0.0))
            evidence_refs = list(raw.get("evidence_refs", []))
            matched = len(evidence_refs) if evidence_refs else evidence_count
            support = min(1.0, evidence_support_base + (matched / max(1, self.config.max_evidence_units)) * 0.25)
            confidence = max(0.0, min(1.0, (raw_confidence * 0.75) + (support * 0.25)))
            disagreement = max(0.0, min(1.0, raw_disagreement + (0.15 if support < self.config.confidence_floor else 0.0)))
            unsupported = confidence < self.config.confidence_floor or support < self.config.confidence_floor
            conflict_prone = disagreement > self.config.disagreement_ceiling or bool(raw.get("conflict_prone", False))
            expanded.append(
                EvidenceGuidedRouteCandidate(
                    route_id=route_id,
                    selected_depth_roles=list(raw.get("selected_depth_roles", [])),
                    confidence=confidence,
                    disagreement=disagreement,
                    evidence_support=support,
                    unsupported=unsupported,
                    conflict_prone=conflict_prone,
                    metadata={
                        "raw_confidence": raw_confidence,
                        "raw_disagreement": raw_disagreement,
                        "evidence_units_seen": evidence_count,
                        "source": "evidence_guided_route_expander",
                    },
                )
            )

        expanded.sort(key=lambda item: (item.conflict_prone, item.unsupported, -item.confidence, item.disagreement))
        report = EvidenceGuidedRouteExpansionReport(
            enabled=True,
            candidates=expanded,
            metadata={"bounded_by": self.config.max_route_candidates, "evidence_units": evidence_count},
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _extract_evidence_units(evidence_report: Optional[Any]) -> List[Dict[str, Any]]:
        if evidence_report is None:
            return []
        if hasattr(evidence_report, "evidence_units"):
            return [unit.to_dict() if hasattr(unit, "to_dict") else _safe_jsonable(unit) for unit in evidence_report.evidence_units]
        if hasattr(evidence_report, "to_dict"):
            payload = evidence_report.to_dict()
        elif isinstance(evidence_report, dict):
            payload = evidence_report
        else:
            return []
        units = payload.get("evidence_units", [])
        return units if isinstance(units, list) else []


def evidence_guided_route_expander_contract() -> Dict[str, Any]:
    return {
        "module": "evidence_guided_route_expander",
        "stage": "REASON-3A",
        "default_enabled": False,
        "evidence_metadata_only": True,
        "permanent_memory_store_mutation": False,
        "bounded_candidates": True,
        "paamax_metadata": True,
    }
