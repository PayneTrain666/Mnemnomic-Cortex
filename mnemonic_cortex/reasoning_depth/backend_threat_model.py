from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class BackendThreatModelError(ValueError):
    """Raised when backend threat model input is unsafe."""


class ThreatSeverity(str, Enum):
    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class BackendThreatModelConfig:
    enabled: bool = False
    max_threats: int = 64
    include_credentials: bool = True
    include_integrity: bool = True
    include_availability: bool = True
    include_privacy: bool = True
    include_migration_risk: bool = True
    require_json_safe_report: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_threats <= 0:
            raise BackendThreatModelError("max_threats must be positive")
        if not self.no_mutation_by_default:
            raise BackendThreatModelError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "BackendThreatModelConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "BackendThreatModelConfig":
        return cls(enabled=True)


@dataclass
class BackendThreatRecord:
    category: str
    threat: str
    severity: ThreatSeverity
    mitigation: str
    verification: str
    threat_id: str = field(default_factory=lambda: f"backend_threat_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "threat_id": self.threat_id,
            "category": self.category,
            "threat": self.threat,
            "severity": self.severity.value,
            "mitigation": self.mitigation,
            "verification": self.verification,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class BackendThreatModelReport:
    enabled: bool
    threats: List[BackendThreatRecord]
    blocked_until_resolved: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"backend_threat_model_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "threats": [threat.to_dict() for threat in self.threats],
            "threat_count": len(self.threats),
            "blocked_until_resolved": list(self.blocked_until_resolved),
            "lineage": _safe_jsonable(self.lineage),
            "real_store_write_authorized": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class BackendThreatModelBuilder:
    """Builds a planning-only threat model for future persistence backend work."""

    def __init__(self, config: Optional[BackendThreatModelConfig] = None):
        self.config = config or BackendThreatModelConfig.disabled()
        self.config.validate()

    def build(self, *, lineage: Optional[Dict[str, Any]] = None) -> BackendThreatModelReport:
        if not self.config.enabled:
            return BackendThreatModelReport(enabled=False, threats=[], blocked_until_resolved=["threat_model_disabled"], lineage=lineage or {})

        threats: List[BackendThreatRecord] = []
        if self.config.include_credentials:
            threats.append(self._record("credentials", "overbroad credentials leak or mutate unintended stores", ThreatSeverity.BLOCKER, "least-privilege scoped credentials; no secrets in traces", "secret scan and permission tests"))
        if self.config.include_integrity:
            threats.append(self._record("integrity", "duplicate or replayed commits corrupt memory lineage", ThreatSeverity.HIGH, "idempotency keys and append-only audit ledger", "duplicate replay tests"))
            threats.append(self._record("integrity", "partial writes leave graph/trace state inconsistent", ThreatSeverity.HIGH, "transaction boundary or two-phase commit plan", "failure-injection tests"))
        if self.config.include_availability:
            threats.append(self._record("availability", "backend outage blocks reasoning controller", ThreatSeverity.MEDIUM, "degrade to dry-run/no-persist mode", "dependency outage test"))
        if self.config.include_privacy:
            threats.append(self._record("privacy", "sensitive trace data persisted without redaction", ThreatSeverity.HIGH, "redaction layer before payload serialization", "redaction regression tests"))
        if self.config.include_migration_risk:
            threats.append(self._record("migration", "schema migration is destructive or irreversible", ThreatSeverity.BLOCKER, "backup + dry-run migration + rollback plan", "migration dry-run and restore test"))

        threats = threats[: self.config.max_threats]
        report = BackendThreatModelReport(
            enabled=True,
            threats=threats,
            blocked_until_resolved=[
                "credential_scope_review",
                "backup_restore_drill",
                "migration_dry_run",
                "redaction_gate",
                "idempotency_gate",
                "explicit_second_authorization",
            ],
            lineage=lineage or {},
        )
        if self.config.require_json_safe_report:
            json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _record(category: str, threat: str, severity: ThreatSeverity, mitigation: str, verification: str) -> BackendThreatRecord:
        return BackendThreatRecord(category=category, threat=threat, severity=severity, mitigation=mitigation, verification=verification)


def backend_threat_model_contract() -> Dict[str, Any]:
    return {
        "module": "backend_threat_model",
        "stage": "FUTURE-BACKEND-AUTHORIZATION",
        "planning_only": True,
        "real_store_write_authorized": False,
        "covers_credentials_integrity_availability_privacy_migration": True,
        "json_safe_report": True,
    }
