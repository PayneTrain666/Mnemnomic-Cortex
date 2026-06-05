"""QSPIN-PROD-8 metadata-only observability/evidence pack."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Mapping, Tuple
import json

class Prod8MetricName(str, Enum):
    FINAL_READINESS_DOMAINS_REVIEWED = "final_readiness_domains_reviewed"
    FINAL_READINESS_DOMAINS_BLOCKED = "final_readiness_domains_blocked"
    PRODUCTION_BLOCKERS_OPEN = "production_blockers_open"
    PRODUCTION_BLOCKERS_P0 = "production_blockers_p0"
    PRODUCTION_BLOCKERS_P1 = "production_blockers_p1"
    CI_BASELINE_GATES_FROZEN = "ci_baseline_gates_frozen"
    CI_BASELINE_GATES_BLOCKED = "ci_baseline_gates_blocked"
    PROBE_REPORT_FINDINGS = "probe_report_findings"
    PROBE_REPORT_CRITICAL_FINDINGS = "probe_report_critical_findings"
    FINAL_REMEDIATION_ITEMS_OPEN = "final_remediation_items_open"
    FINAL_REMEDIATION_ITEMS_DEFERRED = "final_remediation_items_deferred"
    RELEASE_ARTIFACTS_REQUIRED = "release_artifacts_required"
    RELEASE_ARTIFACTS_PRESENT = "release_artifacts_present"
    RAW_PAYLOAD_REJECTIONS = "raw_payload_rejections"
    SECRET_REJECTIONS = "secret_rejections"
    LIVE_ROUTE_REJECTIONS = "live_route_rejections"
    WRITE_REJECTIONS = "write_rejections"
    COMMIT_REJECTIONS = "commit_rejections"
    PRODUCTION_ACTIVATION_REJECTIONS = "production_activation_rejections"
    STRUCTURED_SKIPS = "structured_skips"

@dataclass(frozen=True)
class Prod8MetricEvent:
    name: Prod8MetricName
    value: int
    tags: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> "Prod8MetricEvent":
        if self.value < 0:
            raise ValueError("metric value must be non-negative")
        return self

@dataclass(frozen=True)
class Prod8AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    raw_payload_free: bool = True
    secret_free: bool = True

    def validate(self) -> "Prod8AuditEvent":
        if not self.event_id or not self.action:
            raise ValueError("audit event requires id and action")
        if not self.raw_payload_free or not self.secret_free:
            raise ValueError("unsafe audit event")
        return self

@dataclass(frozen=True)
class Prod8SpanEvent:
    span_id: str
    name: str
    status: str
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> "Prod8SpanEvent":
        if not self.span_id or not self.name:
            raise ValueError("span requires id and name")
        return self

@dataclass(frozen=True)
class Prod8DeadLetterEvent:
    event_id: str
    reason_codes: Tuple[str, ...]
    raw_payload_free: bool = True
    secret_free: bool = True

    def validate(self) -> "Prod8DeadLetterEvent":
        if not self.event_id or not self.reason_codes:
            raise ValueError("dead letter requires id and reason codes")
        if not self.raw_payload_free or not self.secret_free:
            raise ValueError("unsafe dead letter")
        return self

@dataclass(frozen=True)
class Prod8EvidenceRecord:
    evidence_id: str
    kind: str
    safe_summary: Mapping[str, object]

    def validate(self) -> "Prod8EvidenceRecord":
        if not self.evidence_id or not self.kind:
            raise ValueError("evidence requires id and kind")
        return self

@dataclass(frozen=True)
class Prod8DiagnosticSnapshot:
    metrics: Tuple[Prod8MetricEvent, ...]
    audits: Tuple[Prod8AuditEvent, ...]
    spans: Tuple[Prod8SpanEvent, ...]
    dead_letters: Tuple[Prod8DeadLetterEvent, ...]
    evidence: Tuple[Prod8EvidenceRecord, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": [{"name": m.name.value, "value": m.value, "tags": dict(m.tags)} for m in self.metrics],
            "audits": [asdict(a) for a in self.audits],
            "spans": [asdict(s) for s in self.spans],
            "dead_letters": [asdict(d) for d in self.dead_letters],
            "evidence": [asdict(e) for e in self.evidence],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

class Prod8ObservabilityCollector:
    def __init__(self):
        self.metrics = []
        self.audits: Dict[str, Prod8AuditEvent] = {}
        self.spans: Dict[str, Prod8SpanEvent] = {}
        self.dead_letters: Dict[str, Prod8DeadLetterEvent] = {}
        self.evidence: Dict[str, Prod8EvidenceRecord] = {}

    def emit_metric(self, metric: Prod8MetricEvent) -> None:
        self.metrics.append(metric.validate())

    def emit_audit(self, audit: Prod8AuditEvent) -> None:
        audit.validate(); self.audits.setdefault(audit.event_id, audit)

    def emit_span(self, span: Prod8SpanEvent) -> None:
        span.validate(); self.spans.setdefault(span.span_id, span)

    def emit_dead_letter(self, event: Prod8DeadLetterEvent) -> None:
        event.validate(); self.dead_letters.setdefault(event.event_id, event)

    def emit_evidence(self, evidence: Prod8EvidenceRecord) -> None:
        evidence.validate(); self.evidence.setdefault(evidence.evidence_id, evidence)

    def snapshot(self) -> Prod8DiagnosticSnapshot:
        return Prod8DiagnosticSnapshot(tuple(self.metrics), tuple(self.audits.values()), tuple(self.spans.values()), tuple(self.dead_letters.values()), tuple(self.evidence.values()))

def build_default_prod8_observability_collector() -> Prod8ObservabilityCollector:
    return Prod8ObservabilityCollector()
