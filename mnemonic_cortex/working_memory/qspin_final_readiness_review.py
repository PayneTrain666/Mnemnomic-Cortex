"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-8 final pre-activation readiness review.

This module is read-only and pre-activation only. It consolidates evidence from
QD6A, QSPIN-8, and PROD-0 through PROD-7, then deliberately avoids any
production-ready claim while critical blockers remain.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import json

class FinalReadinessMode(str, Enum):
    PRE_ACTIVATION_REVIEW = "pre_activation_review"
    DISABLED = "disabled"

class FinalReadinessStatus(str, Enum):
    REVIEWED_HOLD_REQUIRED = "reviewed_hold_required"
    BLOCKED = "blocked"
    DISABLED = "disabled"

class FinalReadinessBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    MISSING_CRITICAL_EVIDENCE = "missing_critical_evidence"
    CRITICAL_BLOCKERS_OPEN = "critical_blockers_open"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"
    WRITE_PERMISSION_REQUESTED = "write_permission_requested"
    LIVE_ROUTING_REQUESTED = "live_routing_requested"
    PAYLOAD_TRANSFER_REQUESTED = "payload_transfer_requested"
    COMMIT_REQUESTED = "commit_requested"

class FinalReadinessDomain(str, Enum):
    SOURCE_LINEAGE = "source_lineage"
    QD6A_COMPATIBILITY = "qd6a_compatibility"
    QSPIN_PRESERVATION = "qspin_0_through_8_preservation"
    PROD_PRESERVATION = "prod_0_through_7_preservation"
    SAFETY_BOUNDARIES = "safety_boundaries"
    READONLY_PROBE = "readonly_probe_evidence"
    SYNTHETIC_REAL_BOUNDARY = "synthetic_to_real_boundary_evidence"
    CI_GATE = "ci_gate_evidence"
    OBSERVABILITY = "observability_evidence"
    READINESS_BLOCKERS = "readiness_blocker_evidence"
    SECURITY_DATA_SAFETY = "security_data_safety_evidence"
    ROLLBACK = "rollback_evidence"
    KILL_SWITCH = "kill_switch_evidence"
    PRODUCTION_CAVEATS = "production_caveats"
    DEFERRED_HARDENING = "deferred_hardening"
    OPERATOR_APPROVAL = "operator_approval_placeholder"
    EXTERNAL_SECURITY_REVIEW = "external_security_review_placeholder"
    PERFORMANCE_BENCHMARK = "performance_benchmark_placeholder"
    LIVE_CANARY = "live_canary_placeholder"
    LIVE_ROLLBACK_TEST = "live_rollback_test_placeholder"

class FinalReadinessEvidenceKind(str, Enum):
    PRESENT = "present"
    PLACEHOLDER = "placeholder"
    MISSING = "missing"
    DEFERRED = "deferred"

@dataclass(frozen=True)
class FinalReadinessEvidenceRecord:
    domain: FinalReadinessDomain
    kind: FinalReadinessEvidenceKind
    evidence_id: str
    description: str
    critical: bool = True
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> "FinalReadinessEvidenceRecord":
        if not self.evidence_id or not self.description:
            raise ValueError("evidence_id and description are required")
        if not isinstance(self.domain, FinalReadinessDomain):
            raise ValueError("domain must be FinalReadinessDomain")
        if not isinstance(self.kind, FinalReadinessEvidenceKind):
            raise ValueError("kind must be FinalReadinessEvidenceKind")
        return self

@dataclass(frozen=True)
class FinalReadinessDomainResult:
    domain: FinalReadinessDomain
    passed: bool
    critical: bool
    status: str
    evidence_ids: Tuple[str, ...]
    missing_evidence: Tuple[str, ...] = ()
    deferred_items: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "domain": self.domain.value,
            "passed": self.passed,
            "critical": self.critical,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "missing_evidence": list(self.missing_evidence),
            "deferred_items": list(self.deferred_items),
        }

@dataclass(frozen=True)
class FinalReadinessReviewConfig:
    mode: FinalReadinessMode = FinalReadinessMode.PRE_ACTIVATION_REVIEW
    allow_production_ready_claim: bool = False
    allow_activation_command: bool = False
    fail_on_critical_missing: bool = True

    def validate(self) -> "FinalReadinessReviewConfig":
        if self.mode == FinalReadinessMode.DISABLED:
            raise ValueError("final readiness review disabled")
        if self.allow_production_ready_claim:
            raise ValueError("production-ready claim is forbidden in PROD-8")
        if self.allow_activation_command:
            raise ValueError("activation command is forbidden in PROD-8")
        return self

@dataclass(frozen=True)
class FinalReadinessReviewRequest:
    request_id: str
    evidence: Tuple[FinalReadinessEvidenceRecord, ...]
    critical_blockers_open: int
    production_activation_requested: bool = False
    write_permission_requested: bool = False
    live_routing_requested: bool = False
    payload_transfer_requested: bool = False
    commit_requested: bool = False

    def validate(self) -> "FinalReadinessReviewRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        for record in self.evidence:
            record.validate()
        if self.critical_blockers_open < 0:
            raise ValueError("critical_blockers_open must be non-negative")
        return self

@dataclass(frozen=True)
class FinalReadinessReviewResult:
    request_id: str
    status: FinalReadinessStatus
    production_ready: bool
    hold_required: bool
    block_reasons: Tuple[FinalReadinessBlockReason, ...]
    domain_results: Tuple[FinalReadinessDomainResult, ...]
    final_recommendation: str
    missing_evidence: Tuple[str, ...]
    deferred_items: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "production_ready": self.production_ready,
            "hold_required": self.hold_required,
            "block_reasons": [r.value for r in self.block_reasons],
            "domain_results": [d.to_dict() for d in self.domain_results],
            "final_recommendation": self.final_recommendation,
            "missing_evidence": list(self.missing_evidence),
            "deferred_items": list(self.deferred_items),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# PROD-8 Final Pre-Activation Readiness Review", "", f"Status: `{self.status.value}`", f"Production ready: `{self.production_ready}`", f"Hold required: `{self.hold_required}`", "", "## Block reasons"]
        lines.extend(f"- {r.value}" for r in self.block_reasons)
        lines.append("\n## Domains")
        for domain in self.domain_results:
            lines.append(f"- `{domain.domain.value}`: {'PASS' if domain.passed else 'GAP'} ({domain.status})")
        lines.append("\n## Recommendation")
        lines.append(self.final_recommendation)
        return "\n".join(lines) + "\n"

class FinalPreActivationReadinessReviewer:
    def __init__(self, config: FinalReadinessReviewConfig | None = None):
        self.config = (config or build_default_final_readiness_config()).validate()

    def review(self, request: FinalReadinessReviewRequest) -> FinalReadinessReviewResult:
        request.validate()
        reasons: List[FinalReadinessBlockReason] = []
        if request.production_activation_requested:
            reasons.append(FinalReadinessBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if request.write_permission_requested:
            reasons.append(FinalReadinessBlockReason.WRITE_PERMISSION_REQUESTED)
        if request.live_routing_requested:
            reasons.append(FinalReadinessBlockReason.LIVE_ROUTING_REQUESTED)
        if request.payload_transfer_requested:
            reasons.append(FinalReadinessBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.commit_requested:
            reasons.append(FinalReadinessBlockReason.COMMIT_REQUESTED)
        if request.critical_blockers_open > 0:
            reasons.append(FinalReadinessBlockReason.CRITICAL_BLOCKERS_OPEN)

        by_domain: Dict[FinalReadinessDomain, List[FinalReadinessEvidenceRecord]] = {d: [] for d in build_default_final_readiness_domains()}
        for record in request.evidence:
            by_domain.setdefault(record.domain, []).append(record)

        domain_results: List[FinalReadinessDomainResult] = []
        missing: List[str] = []
        deferred: List[str] = []
        for domain in build_default_final_readiness_domains():
            records = by_domain.get(domain, [])
            present = [r for r in records if r.kind == FinalReadinessEvidenceKind.PRESENT]
            placeholders = [r for r in records if r.kind == FinalReadinessEvidenceKind.PLACEHOLDER]
            missing_records = [r for r in records if r.kind == FinalReadinessEvidenceKind.MISSING]
            deferred_records = [r for r in records if r.kind == FinalReadinessEvidenceKind.DEFERRED]
            critical = any(r.critical for r in records) if records else True
            passed = bool(present) and not any(r.critical and r.kind == FinalReadinessEvidenceKind.MISSING for r in records)
            if not records:
                missing.append(domain.value)
                passed = False
            for rec in missing_records:
                missing.append(rec.evidence_id)
            for rec in deferred_records + placeholders:
                deferred.append(rec.evidence_id)
            domain_results.append(FinalReadinessDomainResult(
                domain=domain,
                passed=passed,
                critical=critical,
                status="present" if passed else "missing_or_deferred",
                evidence_ids=tuple(r.evidence_id for r in records),
                missing_evidence=tuple(r.evidence_id for r in missing_records) or (() if records else (domain.value,)),
                deferred_items=tuple(r.evidence_id for r in deferred_records + placeholders),
            ))

        if missing and self.config.fail_on_critical_missing:
            reasons.append(FinalReadinessBlockReason.MISSING_CRITICAL_EVIDENCE)
        reasons = list(dict.fromkeys(reasons))
        status = FinalReadinessStatus.BLOCKED if reasons else FinalReadinessStatus.REVIEWED_HOLD_REQUIRED
        # PROD-8 intentionally never sets production_ready True.
        recommendation = "FINAL HOLD REQUIRED: preserve pre-activation package; do not enable production activation until blockers are burned down under a separate explicit command."
        return FinalReadinessReviewResult(
            request_id=request.request_id,
            status=status,
            production_ready=False,
            hold_required=True,
            block_reasons=tuple(reasons),
            domain_results=tuple(domain_results),
            final_recommendation=recommendation,
            missing_evidence=tuple(missing),
            deferred_items=tuple(deferred),
        )

def build_default_final_readiness_config() -> FinalReadinessReviewConfig:
    return FinalReadinessReviewConfig().validate()

def build_default_final_readiness_domains() -> Tuple[FinalReadinessDomain, ...]:
    return tuple(FinalReadinessDomain)

def build_default_final_readiness_evidence() -> Tuple[FinalReadinessEvidenceRecord, ...]:
    records = []
    placeholder_domains = {
        FinalReadinessDomain.OPERATOR_APPROVAL,
        FinalReadinessDomain.EXTERNAL_SECURITY_REVIEW,
        FinalReadinessDomain.PERFORMANCE_BENCHMARK,
        FinalReadinessDomain.LIVE_CANARY,
        FinalReadinessDomain.LIVE_ROLLBACK_TEST,
    }
    for domain in build_default_final_readiness_domains():
        kind = FinalReadinessEvidenceKind.PLACEHOLDER if domain in placeholder_domains else FinalReadinessEvidenceKind.PRESENT
        records.append(FinalReadinessEvidenceRecord(
            domain=domain,
            kind=kind,
            evidence_id=f"evidence_{domain.value}",
            description=f"PROD-8 readiness evidence for {domain.value}",
            critical=domain not in placeholder_domains,
        ))
    return tuple(records)
