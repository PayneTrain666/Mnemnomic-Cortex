"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-7 local CI gate enforcement.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

class CIGateMode(str, Enum):
    DISABLED = "disabled"
    LOCAL_ENFORCE = "local_enforce"

class CIGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class CIGateBlockReason(str, Enum):
    CRITICAL_GATE_FAILED = "critical_gate_failed"
    SAFETY_BOUNDARY_VIOLATED = "safety_boundary_violated"
    OPTIONAL_TOOL_MISSING = "optional_tool_missing"
    MODE_DISABLED = "mode_disabled"

class CIGateSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    OPTIONAL = "optional"

@dataclass(frozen=True)
class CIGatePolicy:
    mode: CIGateMode = CIGateMode.LOCAL_ENFORCE
    fail_on_critical: bool = True
    fail_on_safety_boundary: bool = True
    allow_optional_skips: bool = True

    def validate(self) -> "CIGatePolicy":
        if self.mode == CIGateMode.DISABLED:
            raise ValueError("CI gate policy disabled")
        return self

@dataclass(frozen=True)
class CIGateCheck:
    check_id: str
    description: str
    severity: CIGateSeverity
    passed: bool = True
    optional: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CIGateCheckResult:
    check_id: str
    status: CIGateStatus
    severity: CIGateSeverity
    reasons: Tuple[CIGateBlockReason, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"check_id": self.check_id, "status": self.status.value, "severity": self.severity.value, "reasons": [r.value for r in self.reasons], "evidence": dict(self.evidence)}

@dataclass(frozen=True)
class CIGateSuiteResult:
    results: Tuple[CIGateCheckResult, ...]

    @property
    def passed(self) -> int: return sum(1 for r in self.results if r.status == CIGateStatus.PASSED)
    @property
    def failed(self) -> int: return sum(1 for r in self.results if r.status == CIGateStatus.FAILED)
    @property
    def skipped(self) -> int: return sum(1 for r in self.results if r.status == CIGateStatus.SKIPPED)
    def to_dict(self) -> Dict[str, Any]: return {"passed": self.passed, "failed": self.failed, "skipped": self.skipped, "results": [r.to_dict() for r in self.results]}
    def to_junit_xml(self) -> str:
        cases = []
        for r in self.results:
            if r.status == CIGateStatus.FAILED:
                cases.append(f'<testcase name="{r.check_id}"><failure>{",".join(x.value for x in r.reasons)}</failure></testcase>')
            elif r.status == CIGateStatus.SKIPPED:
                cases.append(f'<testcase name="{r.check_id}"><skipped /></testcase>')
            else:
                cases.append(f'<testcase name="{r.check_id}" />')
        return f'<testsuite tests="{len(self.results)}" failures="{self.failed}" skipped="{self.skipped}">' + "".join(cases) + "</testsuite>"

class CIGateEnforcer:
    def __init__(self, policy: Optional[CIGatePolicy] = None):
        self.policy = (policy or build_default_ci_gate_policy()).validate()

    def run(self, checks: Iterable[CIGateCheck]) -> CIGateSuiteResult:
        results = []
        for c in checks:
            if c.optional and not c.passed and self.policy.allow_optional_skips:
                results.append(CIGateCheckResult(c.check_id, CIGateStatus.SKIPPED, c.severity, (CIGateBlockReason.OPTIONAL_TOOL_MISSING,), c.evidence))
            elif not c.passed:
                reason = CIGateBlockReason.SAFETY_BOUNDARY_VIOLATED if c.severity in (CIGateSeverity.CRITICAL, CIGateSeverity.HIGH) else CIGateBlockReason.CRITICAL_GATE_FAILED
                results.append(CIGateCheckResult(c.check_id, CIGateStatus.FAILED, c.severity, (reason,), c.evidence))
            else:
                results.append(CIGateCheckResult(c.check_id, CIGateStatus.PASSED, c.severity, evidence=c.evidence))
        return CIGateSuiteResult(tuple(results))

def build_default_ci_gate_policy() -> CIGatePolicy:
    return CIGatePolicy().validate()

def build_default_ci_gate_checks() -> Tuple[CIGateCheck, ...]:
    critical = CIGateSeverity.CRITICAL
    optional = CIGateSeverity.OPTIONAL
    names = [
        "source_verification_gate", "token_budget_gate", "source_consideration_matrix_gate", "no_live_routing_gate",
        "no_payload_transfer_gate", "no_real_write_gate", "no_qh_write_gate", "no_shared_slot_write_gate",
        "no_external_memory_write_gate", "no_commit_gate", "no_production_activation_gate", "redaction_gate",
        "audit_chain_gate", "deterministic_replay_gate", "trace_corpus_replay_gate", "ci_matrix_hardening_gate",
        "extended_safety_regression_gate", "remediation_closure_gate", "full_printout_gate", "ship_check_gate"
    ]
    return tuple(CIGateCheck(n, n.replace("_", " "), critical, True) for n in names) + (CIGateCheck("pytest_optional", "pytest optional availability", optional, False, True),)
