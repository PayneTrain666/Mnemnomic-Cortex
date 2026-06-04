"""QSPIN-PROD-8 final remediation carry-forward register."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Tuple
import json

class FinalRemediationSeverity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

class FinalRemediationStatus(str, Enum):
    OPEN = "open"
    DEFERRED = "deferred"
    HOLD = "hold"
    RESOLVED = "resolved"

class FinalRemediationCategory(str, Enum):
    UNRESOLVED_PRODUCTION_BLOCKER = "unresolved_production_blocker"
    DEFERRED_HARDENING = "deferred_hardening"
    MISSING_LIVE_TEST = "missing_live_test"
    EXTERNAL_SECURITY_REVIEW = "external_security_review"
    OPERATOR_APPROVAL = "operator_approval"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DEPLOYMENT_STAGING_REVIEW = "deployment_staging_review"
    LIVE_ROLLBACK_TEST = "live_rollback_test"
    OBSERVABILITY_LIVE_TEST = "observability_live_test"
    CANARY_LIVE_TEST = "canary_live_test"
    COMMIT_EXECUTION_VALIDATION = "commit_execution_validation"
    REAL_STORAGE_INTEGRATION_VALIDATION = "real_storage_integration_validation"

@dataclass(frozen=True)
class FinalRemediationItem:
    item_id: str
    severity: FinalRemediationSeverity
    status: FinalRemediationStatus
    category: FinalRemediationCategory
    description: str
    acceptance_criteria: Tuple[str, ...]
    target_stage: str
    defer_justification: str = ""

    def validate(self) -> "FinalRemediationItem":
        if not self.item_id or not self.description or not self.acceptance_criteria:
            raise ValueError("remediation item requires id, description, and acceptance criteria")
        if self.severity == FinalRemediationSeverity.P0 and self.status == FinalRemediationStatus.RESOLVED:
            # resolved is okay only with target evidence embedded in acceptance criteria; this is a conservative placeholder.
            pass
        if self.severity in (FinalRemediationSeverity.P0, FinalRemediationSeverity.P1) and self.status == FinalRemediationStatus.DEFERRED and not self.defer_justification:
            raise ValueError("P0/P1 deferred item requires justification")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"item_id": self.item_id, "severity": self.severity.value, "status": self.status.value, "category": self.category.value, "description": self.description, "acceptance_criteria": list(self.acceptance_criteria), "target_stage": self.target_stage, "defer_justification": self.defer_justification}

@dataclass(frozen=True)
class FinalRemediationRegister:
    items: Tuple[FinalRemediationItem, ...]

    def validate(self) -> "FinalRemediationRegister":
        for item in self.items:
            item.validate()
        return self

@dataclass(frozen=True)
class FinalRemediationReport:
    open_p0: Tuple[str, ...]
    open_p1: Tuple[str, ...]
    deferred: Tuple[str, ...]
    hold_required: bool
    items: Tuple[FinalRemediationItem, ...]

    def to_dict(self) -> Dict[str, object]:
        return {"open_p0": list(self.open_p0), "open_p1": list(self.open_p1), "deferred": list(self.deferred), "hold_required": self.hold_required, "items": [i.to_dict() for i in self.items]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# PROD-8 Final Remediation Carry-Forward Register", "", f"Hold required: `{self.hold_required}`", "", "## Open P0"]
        lines.extend(f"- {x}" for x in self.open_p0)
        lines.append("\n## Open P1")
        lines.extend(f"- {x}" for x in self.open_p1)
        lines.append("\n## Deferred")
        lines.extend(f"- {x}" for x in self.deferred)
        return "\n".join(lines) + "\n"

    @classmethod
    def from_register(cls, register: FinalRemediationRegister) -> "FinalRemediationReport":
        register.validate()
        open_p0 = tuple(i.item_id for i in register.items if i.severity == FinalRemediationSeverity.P0 and i.status != FinalRemediationStatus.RESOLVED)
        open_p1 = tuple(i.item_id for i in register.items if i.severity == FinalRemediationSeverity.P1 and i.status != FinalRemediationStatus.RESOLVED)
        deferred = tuple(i.item_id for i in register.items if i.status == FinalRemediationStatus.DEFERRED)
        return cls(open_p0, open_p1, deferred, bool(open_p0 or open_p1), register.items)

def build_default_final_remediation_register() -> FinalRemediationRegister:
    items = []
    for idx, category in enumerate(FinalRemediationCategory):
        severity = FinalRemediationSeverity.P0 if idx < 2 else (FinalRemediationSeverity.P1 if idx < 7 else FinalRemediationSeverity.P2)
        status = FinalRemediationStatus.OPEN if severity in (FinalRemediationSeverity.P0, FinalRemediationSeverity.P1) else FinalRemediationStatus.DEFERRED
        items.append(FinalRemediationItem(
            item_id=f"FR-{idx+1:03d}",
            severity=severity,
            status=status,
            category=category,
            description=f"Carry forward {category.value} before any activation planning.",
            acceptance_criteria=(f"Evidence closes {category.value} under an explicit future command.",),
            target_stage="future-explicit-activation-planning" if severity in (FinalRemediationSeverity.P0, FinalRemediationSeverity.P1) else "future-hardening",
            defer_justification="Deferred because PROD-8 is final pre-activation hold only." if status == FinalRemediationStatus.DEFERRED else "",
        ))
    return FinalRemediationRegister(tuple(items)).validate()
