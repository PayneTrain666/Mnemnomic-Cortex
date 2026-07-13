"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-7 metadata-only observability collector.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

class Prod7MetricName(str, Enum):
    READONLY_PROBE_ATTEMPTS = "readonly_probe_attempts"
    READONLY_PROBE_PASSED = "readonly_probe_passed"
    READONLY_PROBE_FAILED = "readonly_probe_failed"
    READONLY_PROBE_SKIPPED = "readonly_probe_skipped"
    BOUNDARY_VERIFICATION_ATTEMPTS = "boundary_verification_attempts"
    BOUNDARY_VERIFICATION_BLOCKED = "boundary_verification_blocked"
    CI_GATE_CHECKS_RUN = "ci_gate_checks_run"
    CI_GATE_CHECKS_FAILED = "ci_gate_checks_failed"
    OBSERVABILITY_SIGNALS_REVIEWED = "observability_signals_reviewed"
    OBSERVABILITY_GAPS_FOUND = "observability_gaps_found"
    READINESS_BLOCKERS_OPEN = "readiness_blockers_open"
    READINESS_BLOCKERS_CRITICAL = "readiness_blockers_critical"
    PROBE_SAFETY_CASES_RUN = "probe_safety_cases_run"
    PROBE_SAFETY_CASES_FAILED = "probe_safety_cases_failed"
    RAW_PAYLOAD_REJECTIONS = "raw_payload_rejections"
    SECRET_REJECTIONS = "secret_rejections"
    LIVE_ROUTE_REJECTIONS = "live_route_rejections"
    WRITE_REJECTIONS = "write_rejections"
    COMMIT_REJECTIONS = "commit_rejections"
    PRODUCTION_ACTIVATION_REJECTIONS = "production_activation_rejections"
    STRUCTURED_SKIPS = "structured_skips"

@dataclass(frozen=True)
class Prod7MetricEvent:
    name: Prod7MetricName
    value: int = 1
    tags: Mapping[str, str] = field(default_factory=dict)
    def validate(self):
        if self.value < 0: raise ValueError("metric value must be non-negative")
        return self

@dataclass(frozen=True)
class Prod7AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    secret_free: bool = True
    raw_payload_free: bool = True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit event requires id/action")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe audit event")
        return self

@dataclass(frozen=True)
class Prod7SpanEvent:
    span_id: str
    name: str
    safe_summary: Mapping[str, Any] = field(default_factory=dict)
    contains_secret: bool = False
    contains_raw_payload: bool = False
    def validate(self):
        if self.contains_secret or self.contains_raw_payload: raise ValueError("unsafe span")
        return self

@dataclass(frozen=True)
class Prod7DeadLetterEvent:
    event_id: str
    reason_codes: Tuple[str, ...]
    safe_context: Mapping[str, Any] = field(default_factory=dict)
    contains_secret: bool = False
    contains_raw_payload: bool = False
    def validate(self):
        if not self.reason_codes: raise ValueError("dead letter requires reason codes")
        if self.contains_secret or self.contains_raw_payload: raise ValueError("unsafe dead letter")
        return self

@dataclass(frozen=True)
class Prod7DiagnosticSnapshot:
    metrics: Tuple[Prod7MetricEvent, ...]
    audits: Tuple[Prod7AuditEvent, ...]
    spans: Tuple[Prod7SpanEvent, ...]
    dead_letters: Tuple[Prod7DeadLetterEvent, ...]
    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": [{"name": m.name.value, "value": m.value, "tags": dict(m.tags)} for m in self.metrics], "audits": [{"event_id": a.event_id, "action": a.action, "reason_codes": list(a.reason_codes)} for a in self.audits], "spans": [{"span_id": s.span_id, "name": s.name, "safe_summary": dict(s.safe_summary)} for s in self.spans], "dead_letters": [{"event_id": d.event_id, "reason_codes": list(d.reason_codes), "safe_context": dict(d.safe_context)} for d in self.dead_letters]}

class Prod7ObservabilityCollector:
    def __init__(self):
        self.metrics = []
        self.audits = {}
        self.spans = {}
        self.dead_letters = {}
    def emit_metric(self, metric: Prod7MetricEvent): self.metrics.append(metric.validate())
    def emit_audit(self, audit: Prod7AuditEvent): self.audits.setdefault(audit.event_id, audit.validate())
    def emit_span(self, span: Prod7SpanEvent): self.spans.setdefault(span.span_id, span.validate())
    def dead_letter(self, event: Prod7DeadLetterEvent): self.dead_letters.setdefault(event.event_id, event.validate())
    def snapshot(self) -> Prod7DiagnosticSnapshot: return Prod7DiagnosticSnapshot(tuple(self.metrics), tuple(self.audits.values()), tuple(self.spans.values()), tuple(self.dead_letters.values()))

def build_default_prod7_observability_collector() -> Prod7ObservabilityCollector:
    return Prod7ObservabilityCollector()
