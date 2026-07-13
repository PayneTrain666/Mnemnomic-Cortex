"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-7 production-readiness blocker register.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Tuple

class ReadinessBlockerSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ReadinessBlockerStatus(str, Enum):
    OPEN = "open"
    DEFERRED = "deferred"
    RESOLVED = "resolved"

class ReadinessBlockerCategory(str, Enum):
    LIVE_ROUTING_NOT_ENABLED = "live-routing-not-enabled"
    REAL_PAYLOAD_TRANSFER_NOT_ENABLED = "real-payload-transfer-not-enabled"
    REAL_WRITE_PATH_NOT_ENABLED = "real-write-path-not-enabled"
    QH_REAL_STORE_NOT_ENABLED = "qh-real-store-not-enabled"
    SHARED_SLOT_REAL_STORE_NOT_ENABLED = "shared-slot-real-store-not-enabled"
    EXTERNAL_MEMORY_REAL_STORE_NOT_ENABLED = "external-memory-real-store-not-enabled"
    COMMIT_EXECUTION_NOT_ENABLED = "commit-execution-not-enabled"
    PRODUCTION_ACTIVATION_NOT_ENABLED = "production-activation-not-enabled"
    REAL_RUNTIME_INTEGRATION_NOT_VERIFIED = "real-runtime-integration-not-verified"
    PERFORMANCE_BENCHMARKS_NOT_REAL = "performance-benchmarks-not-real"
    SECURITY_REVIEW_NOT_COMPLETE = "security-review-not-complete"
    HAZOP_LOPA_EQUIVALENT_NOT_COMPLETE = "hazop-lopa-equivalent-not-complete"
    OPERATOR_APPROVAL_NOT_COMPLETE = "operator-approval-not-complete"
    ROLLBACK_LIVE_TEST_NOT_COMPLETE = "rollback-live-test-not-complete"
    OBSERVABILITY_LIVE_TEST_NOT_COMPLETE = "observability-live-test-not-complete"
    CANARY_LIVE_TEST_NOT_COMPLETE = "canary-live-test-not-complete"

@dataclass(frozen=True)
class ReadinessBlockerRecord:
    blocker_id: str
    category: ReadinessBlockerCategory
    severity: ReadinessBlockerSeverity
    status: ReadinessBlockerStatus = ReadinessBlockerStatus.OPEN
    evidence: str = "not yet production-validated"
    required_next_action: str = "carry forward to later explicit activation planning"
    target_stage: str = "PROD-8+"
    safety_implication: str = "prevents production-active claim"
    test_coverage: str = "synthetic/read-only only"

    def to_dict(self) -> Dict[str, Any]:
        return {"blocker_id": self.blocker_id, "category": self.category.value, "severity": self.severity.value, "status": self.status.value, "evidence": self.evidence, "required_next_action": self.required_next_action, "target_stage": self.target_stage, "safety_implication": self.safety_implication, "test_coverage": self.test_coverage}

@dataclass(frozen=True)
class ReadinessBlockerReport:
    blockers: Tuple[ReadinessBlockerRecord, ...]

    @property
    def open_critical(self) -> int:
        return sum(1 for b in self.blockers if b.status == ReadinessBlockerStatus.OPEN and b.severity == ReadinessBlockerSeverity.CRITICAL)

    @property
    def production_active_allowed(self) -> bool:
        return self.open_critical == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"production_active_allowed": self.production_active_allowed, "open_critical": self.open_critical, "blockers": [b.to_dict() for b in self.blockers]}

class ReadinessBlockerRegister:
    def __init__(self, blockers: Iterable[ReadinessBlockerRecord] = ()): self.blockers = list(blockers)
    def add(self, blocker: ReadinessBlockerRecord): self.blockers.append(blocker)
    def report(self) -> ReadinessBlockerReport: return ReadinessBlockerReport(tuple(self.blockers))

def build_default_readiness_blocker_register() -> ReadinessBlockerRegister:
    reg = ReadinessBlockerRegister()
    critical = {
        ReadinessBlockerCategory.LIVE_ROUTING_NOT_ENABLED,
        ReadinessBlockerCategory.REAL_PAYLOAD_TRANSFER_NOT_ENABLED,
        ReadinessBlockerCategory.REAL_WRITE_PATH_NOT_ENABLED,
        ReadinessBlockerCategory.COMMIT_EXECUTION_NOT_ENABLED,
        ReadinessBlockerCategory.PRODUCTION_ACTIVATION_NOT_ENABLED,
        ReadinessBlockerCategory.REAL_RUNTIME_INTEGRATION_NOT_VERIFIED,
    }
    for i, cat in enumerate(ReadinessBlockerCategory, 1):
        sev = ReadinessBlockerSeverity.CRITICAL if cat in critical else ReadinessBlockerSeverity.HIGH
        reg.add(ReadinessBlockerRecord(f"RB-{i:03d}", cat, sev))
    return reg
