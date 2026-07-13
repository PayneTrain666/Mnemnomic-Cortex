"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-6 remediation closure workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Tuple

class RemediationClosureStatus(str, Enum):
    RESOLVED = "resolved"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    OPEN = "open"

class RemediationClosureDecision(str, Enum):
    SHIP_ALLOWED = "ship_allowed"
    SHIP_BLOCKED = "ship_blocked"

@dataclass(frozen=True)
class RemediationClosureEvidence:
    evidence_id: str
    description: str
    test_refs: Tuple[str, ...] = ()

    def validate(self) -> "RemediationClosureEvidence":
        if not self.evidence_id or not self.description:
            raise ValueError("closure evidence requires id and description")
        return self

@dataclass(frozen=True)
class RemediationClosureRecord:
    item_id: str
    severity: str
    status: RemediationClosureStatus
    evidence: Tuple[RemediationClosureEvidence, ...] = ()
    defer_target: str = ""
    defer_justification: str = ""
    safety_implication: str = ""

    def validate(self) -> "RemediationClosureRecord":
        if not self.item_id:
            raise ValueError("item_id required")
        if self.severity not in {"P0", "P1", "P2", "P3"}:
            raise ValueError("invalid severity")
        if self.status is RemediationClosureStatus.RESOLVED and not self.evidence:
            raise ValueError("resolved remediation requires evidence")
        if self.status is RemediationClosureStatus.DEFERRED and (not self.defer_target or not self.defer_justification):
            raise ValueError("deferred remediation requires target and justification")
        for e in self.evidence:
            e.validate()
        return self

@dataclass(frozen=True)
class RemediationClosureReport:
    decision: RemediationClosureDecision
    records: Tuple[RemediationClosureRecord, ...]
    prod7_carry_forward: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.decision is RemediationClosureDecision.SHIP_ALLOWED

    def to_json_dict(self) -> Dict[str, object]:
        return {"decision": self.decision.value, "passed": self.passed, "prod7_carry_forward": list(self.prod7_carry_forward), "records": [{"item_id": r.item_id, "severity": r.severity, "status": r.status.value, "defer_target": r.defer_target, "defer_justification": r.defer_justification, "safety_implication": r.safety_implication} for r in self.records]}

    def to_markdown(self) -> str:
        lines = ["# QSPIN-PROD-6 Remediation Closure", "", f"Decision: `{self.decision.value}`", "", "| Item | Severity | Status | Carry Forward |", "|---|---|---|---|"]
        for r in self.records:
            lines.append(f"| {r.item_id} | {r.severity} | {r.status.value} | {r.item_id in self.prod7_carry_forward} |")
        return "\n".join(lines) + "\n"

@dataclass(frozen=True)
class RemediationClosureConfig:
    block_unresolved_p0: bool = True
    require_p1_defer_justification: bool = True

class RemediationClosureWorkflow:
    def __init__(self, config: RemediationClosureConfig | None = None):
        self.config = config or build_default_remediation_closure_config()

    def close(self, records: Tuple[RemediationClosureRecord, ...] | None = None) -> RemediationClosureReport:
        records = records or self.default_records()
        for r in records:
            r.validate()
        blocked = []
        carry = []
        for r in records:
            if r.severity == "P0" and r.status not in {RemediationClosureStatus.RESOLVED, RemediationClosureStatus.SKIPPED}:
                blocked.append(r.item_id)
            if r.severity == "P1" and r.status is RemediationClosureStatus.DEFERRED and not r.defer_justification:
                blocked.append(r.item_id)
            if r.status in {RemediationClosureStatus.DEFERRED, RemediationClosureStatus.BLOCKED, RemediationClosureStatus.OPEN}:
                carry.append(r.item_id)
        decision = RemediationClosureDecision.SHIP_BLOCKED if blocked else RemediationClosureDecision.SHIP_ALLOWED
        return RemediationClosureReport(decision, records, tuple(carry))

    def default_records(self) -> Tuple[RemediationClosureRecord, ...]:
        evidence = (RemediationClosureEvidence("prod6_tests", "PROD-6 synthetic safety tests passed", ("prod6_manual_runner",)),)
        return (
            RemediationClosureRecord("P2_trace_corpus_expand", "P2", RemediationClosureStatus.RESOLVED, evidence, safety_implication="coverage increased"),
            RemediationClosureRecord("P3_ci_matrix_provider", "P3", RemediationClosureStatus.DEFERRED, (), "PROD-7", "real CI provider integration remains out of scope", "no live provider calls in PROD-6"),
        )


def build_default_remediation_closure_config() -> RemediationClosureConfig:
    return RemediationClosureConfig()
