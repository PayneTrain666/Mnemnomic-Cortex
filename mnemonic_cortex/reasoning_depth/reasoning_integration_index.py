"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning integration index.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class ReasoningIntegrationIndexError(ValueError):
    """Raised when integration index construction is unsafe."""


@dataclass(frozen=True)
class ReasoningIntegrationIndexConfig:
    enabled: bool = False
    include_stage_1: bool = True
    include_stage_2: bool = True
    include_stage_3: bool = True
    include_stage_4: bool = True
    require_json_safe_index: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise ReasoningIntegrationIndexError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "ReasoningIntegrationIndexConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningIntegrationIndexConfig":
        return cls(enabled=True)


@dataclass
class IntegrationIndexEntry:
    stage: str
    module: str
    purpose: str
    status: str
    safety_notes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_id: str = field(default_factory=lambda: f"integration_index_entry_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "entry_id": self.entry_id,
            "stage": self.stage,
            "module": self.module,
            "purpose": self.purpose,
            "status": self.status,
            "safety_notes": list(self.safety_notes),
            "dependencies": list(self.dependencies),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class ReasoningIntegrationIndexReport:
    enabled: bool
    entries: List[IntegrationIndexEntry]
    summary: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"reasoning_integration_index_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        entry_payloads = [entry.to_dict() for entry in self.entries]
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "entries": entry_payloads,
            "summary": _safe_jsonable(self.summary),
            "lineage": _safe_jsonable(self.lineage),
            "entry_count": len(entry_payloads),
            "json_safe": True,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningIntegrationIndex:
    """Builds a JSON-safe index of reasoning-depth implementation stages."""

    def __init__(self, config: Optional[ReasoningIntegrationIndexConfig] = None):
        self.config = config or ReasoningIntegrationIndexConfig.disabled()
        self.config.validate()

    def build(self, *, lineage: Optional[Dict[str, Any]] = None) -> ReasoningIntegrationIndexReport:
        if not self.config.enabled:
            return ReasoningIntegrationIndexReport(enabled=False, entries=[], summary={"status": "disabled"}, lineage=lineage or {})

        entries: List[IntegrationIndexEntry] = []
        if self.config.include_stage_1:
            entries.extend(self._stage_1_entries())
        if self.config.include_stage_2:
            entries.extend(self._stage_2_entries())
        if self.config.include_stage_3:
            entries.extend(self._stage_3_entries())
        if self.config.include_stage_4:
            entries.extend(self._stage_4_entries())

        summary = {
            "total_entries": len(entries),
            "stage_1_depth_lattice": self.config.include_stage_1,
            "stage_2_controller_reasoning": self.config.include_stage_2,
            "stage_3_planner_quality": self.config.include_stage_3,
            "stage_4_persistence_metadata": self.config.include_stage_4,
            "automatic_persistence": False,
            "permanent_memory_store_mutation": False,
        }
        report = ReasoningIntegrationIndexReport(enabled=True, entries=entries, summary=summary, lineage=lineage or {})
        if self.config.require_json_safe_index:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _entry(stage: str, module: str, purpose: str, status: str, dependencies: Optional[List[str]] = None) -> IntegrationIndexEntry:
        return IntegrationIndexEntry(
            stage=stage,
            module=module,
            purpose=purpose,
            status=status,
            dependencies=dependencies or [],
            safety_notes=[
                "disabled_by_default_where_runtime_relevant",
                "no_permanent_memory_store_mutation_by_default",
                "json_safe_reports",
            ],
        )

    def _stage_1_entries(self) -> List[IntegrationIndexEntry]:
        return [
            self._entry("REASON-1A", "DepthIndexedSlotLattice", "slots x 8 depth capacity foundation", "complete"),
            self._entry("REASON-1B", "WMDepthController", "optional working-memory depth integration", "complete", ["REASON-1A"]),
            self._entry("REASON-1C", "MANNDepthAdapter", "optional MANN SlotKV depth bank integration", "complete", ["REASON-1A"]),
            self._entry("REASON-1D", "LTMDepthAdapter", "optional LTM depth banks and registry strengthening", "complete", ["REASON-1A"]),
            self._entry("REASON-1E", "DepthCapacityValidation", "capacity validation and readiness smoke benchmarks", "complete", ["REASON-1B", "REASON-1C", "REASON-1D"]),
        ]

    def _stage_2_entries(self) -> List[IntegrationIndexEntry]:
        return [
            self._entry("REASON-2A", "ReasoningController", "WM/MANN/LTM orchestration and shadow consolidation gate", "complete"),
            self._entry("REASON-2B", "ReasoningPolicyRouter", "policy routing, depth-route strategy, confidence/disagreement scoring", "complete", ["REASON-2A"]),
            self._entry("REASON-2C", "EvidenceReasoningPass", "evidence/counterfactual/conflict-aware metadata passes", "complete", ["REASON-2B"]),
            self._entry("REASON-2D", "ReasoningControllerAPI", "API hardening, release audit, regression matrix", "complete", ["REASON-2C"]),
        ]

    def _stage_3_entries(self) -> List[IntegrationIndexEntry]:
        return [
            self._entry("REASON-3A", "MultiPassThoughtPlanner", "strategy graph, multi-pass planner, evidence-guided expansion", "complete", ["REASON-2D"]),
            self._entry("REASON-3B", "PlannerEvaluator", "planner evaluation, failure classification, remediation guidance", "complete", ["REASON-3A"]),
            self._entry("REASON-3C", "ControllerPlannerIntegration", "planner quality hardening and persistence readiness", "complete", ["REASON-3B"]),
            self._entry("REASON-3D", "ReasoningReleaseCandidate", "release candidate hardening and API freeze", "complete", ["REASON-3C"]),
        ]

    def _stage_4_entries(self) -> List[IntegrationIndexEntry]:
        return [
            self._entry("REASON-4A", "ReasoningPersistenceAdapter", "metadata-only persistence payload and commit interface", "complete", ["REASON-3D"]),
            self._entry("REASON-4B", "PersistenceBackendStub", "dry-run backend stubs, dry-run ledger, recovery metadata", "complete", ["REASON-4A"]),
            self._entry("REASON-4C", "PersistenceLineClosure", "final persistence line closure and safety audit", "complete", ["REASON-4B"]),
        ]


def reasoning_integration_index_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_integration_index",
        "stage": "REASON-4C",
        "default_enabled": False,
        "json_safe_index": True,
        "covers_reason_1a_to_4c": True,
        "automatic_persistence": False,
        "permanent_memory_store_mutation": False,
    }
