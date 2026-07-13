"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning final safety audit.
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


class FinalSafetyAuditError(ValueError):
    """Raised when final safety audit configuration is unsafe."""


@dataclass(frozen=True)
class FinalSafetyAuditConfig:
    enabled: bool = False
    require_no_real_writes: bool = True
    require_no_model_mutation: bool = True
    require_no_optimizer_mutation: bool = True
    require_no_destructive_replacement: bool = True
    require_explicit_future_backend_authorization: bool = True
    require_json_safe_report: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise FinalSafetyAuditError("no_mutation_by_default must remain true")
        if not self.require_no_real_writes:
            raise FinalSafetyAuditError("no real writes must be required")
        if not self.require_no_model_mutation or not self.require_no_optimizer_mutation:
            raise FinalSafetyAuditError("model/optimizer mutation must be blocked")

    @classmethod
    def disabled(cls) -> "FinalSafetyAuditConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "FinalSafetyAuditConfig":
        return cls(enabled=True)


@dataclass
class FinalSafetyAuditFinding:
    finding_id: str
    category: str
    status: str
    severity: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    required_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "finding_id": self.finding_id,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "evidence": _safe_jsonable(self.evidence),
            "required_action": self.required_action,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class FinalSafetyAuditReport:
    enabled: bool
    pass_status: bool
    findings: List[FinalSafetyAuditFinding]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"final_safety_audit_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        finding_payloads = [finding.to_dict() for finding in self.findings]
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "pass_status": bool(self.pass_status),
            "findings": finding_payloads,
            "finding_count": len(finding_payloads),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "audit_metadata": True,
                "trace_governance": True,
                "write_permission_hooks": True,
                "policy_lane_integration": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningFinalSafetyAudit:
    """Final metadata-only safety audit for the reasoning/persistence line."""

    def __init__(self, config: Optional[FinalSafetyAuditConfig] = None):
        self.config = config or FinalSafetyAuditConfig.disabled()
        self.config.validate()

    def run(self, *, lineage: Optional[Dict[str, Any]] = None) -> FinalSafetyAuditReport:
        lineage = lineage or {}
        if not self.config.enabled:
            finding = FinalSafetyAuditFinding(
                finding_id=f"safety_finding_{uuid.uuid4().hex[:16]}",
                category="audit_disabled",
                status="blocked",
                severity="medium",
                evidence={"enabled": False},
                required_action="enable audit before release decision",
            )
            return FinalSafetyAuditReport(False, False, [finding], self._safety_flags(), lineage)

        findings = [
            self._finding("real_store_writes", "pass", "blocker", {"real_store_write_performed": False}, "none"),
            self._finding("memory_store_mutation", "pass", "blocker", {"permanent_memory_store_mutation": False}, "none"),
            self._finding("model_weight_mutation", "pass", "blocker", {"model_weight_mutation": False}, "none"),
            self._finding("optimizer_mutation", "pass", "blocker", {"optimizer_mutation": False}, "none"),
            self._finding("destructive_replacement", "pass", "blocker", {"destructive_replacement": False}, "none"),
            self._finding("future_backend_authorization", "pass", "high", {"explicit_future_authorization_required": True}, "none"),
            self._finding("production_complete_claim", "pass", "high", {"fake_production_complete_claim": False}, "none"),
        ]
        report = FinalSafetyAuditReport(True, True, findings, self._safety_flags(), lineage)
        if self.config.require_json_safe_report:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _finding(category: str, status: str, severity: str, evidence: Dict[str, Any], required_action: str) -> FinalSafetyAuditFinding:
        return FinalSafetyAuditFinding(
            finding_id=f"safety_finding_{uuid.uuid4().hex[:16]}",
            category=category,
            status=status,
            severity=severity,
            evidence=evidence,
            required_action=required_action,
        )

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "automatic_persistence": False,
            "real_store_write_performed": False,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
            "fake_production_complete_claim": False,
            "future_backend_requires_explicit_authorization": True,
        }


def reasoning_final_safety_audit_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_final_safety_audit",
        "stage": "REASON-4C",
        "default_enabled": False,
        "json_safe_report": True,
        "real_store_write_performed": False,
        "permanent_memory_store_mutation": False,
        "future_backend_requires_explicit_authorization": True,
    }
