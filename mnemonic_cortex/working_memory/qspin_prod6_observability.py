"""QSPIN-PROD-6 stress observability pack."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Tuple

class Prod6MetricName(str, Enum):
    STRESS_REPLAY_ATTEMPTS = "stress_replay_attempts"
    STRESS_REPLAY_PASSED = "stress_replay_passed"
    STRESS_REPLAY_FAILED = "stress_replay_failed"
    STRESS_REPLAY_SKIPPED = "stress_replay_skipped"
    TRACE_CORPUS_RECORDS_GENERATED = "trace_corpus_records_generated"
    TRACE_CORPUS_REPLAY_PASSED = "trace_corpus_replay_passed"
    TRACE_CORPUS_REPLAY_FAILED = "trace_corpus_replay_failed"
    CI_MATRIX_CASES_RUN = "ci_matrix_cases_run"
    CI_MATRIX_CASES_FAILED = "ci_matrix_cases_failed"
    EXTENDED_SAFETY_CASES_RUN = "extended_safety_cases_run"
    EXTENDED_SAFETY_CASES_FAILED = "extended_safety_cases_failed"
    REMEDIATION_ITEMS_CLOSED = "remediation_items_closed"
    REMEDIATION_ITEMS_DEFERRED = "remediation_items_deferred"
    REMEDIATION_ITEMS_BLOCKED = "remediation_items_blocked"
    RAW_PAYLOAD_REJECTIONS = "raw_payload_rejections"
    SECRET_REJECTIONS = "secret_rejections"
    LIVE_ROUTE_REJECTIONS = "live_route_rejections"
    WRITE_REJECTIONS = "write_rejections"
    COMMIT_REJECTIONS = "commit_rejections"
    PRODUCTION_ACTIVATION_REJECTIONS = "production_activation_rejections"
    DETERMINISTIC_REPLAY_MISMATCHES = "deterministic_replay_mismatches"

@dataclass(frozen=True)
class Prod6MetricEvent:
    name: Prod6MetricName
    value: int = 1
    tags: Mapping[str, str] = field(default_factory=dict)

    def validate(self):
        if self.value < 0:
            raise ValueError("metric value must be non-negative")
        return self

@dataclass(frozen=True)
class Prod6AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    secret_free: bool = True
    raw_payload_free: bool = True

    def validate(self):
        if not self.event_id or not self.action:
            raise ValueError("audit event requires id/action")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("unsafe audit event")
        return self

@dataclass(frozen=True)
class Prod6SpanEvent:
    span_id: str
    name: str
    status: str = "ok"
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def validate(self):
        lowered = str(dict(self.safe_summary)).lower()
        if "raw_payload" in lowered or "secret" in lowered:
            raise ValueError("unsafe span summary")
        return self

@dataclass(frozen=True)
class Prod6DeadLetterEvent:
    record_id: str
    reason_codes: Tuple[str, ...]
    safe_context: Mapping[str, object] = field(default_factory=dict)

    def validate(self):
        if not self.record_id or not self.reason_codes:
            raise ValueError("dead letter requires id/reasons")
        lowered = str(dict(self.safe_context)).lower()
        if "raw_payload" in lowered or "secret" in lowered:
            raise ValueError("unsafe dead letter context")
        return self


@dataclass(frozen=True)
class Prod6TraceRecord:
    trace_id: str
    safe_summary: Mapping[str, object]
    contains_raw_payload: bool = False
    contains_secret: bool = False

    def validate(self):
        if not self.trace_id:
            raise ValueError("trace_id required")
        lowered = str(dict(self.safe_summary)).lower()
        if self.contains_raw_payload or self.contains_secret or "raw_payload" in lowered or "secret" in lowered:
            raise ValueError("unsafe trace record")
        return self

@dataclass(frozen=True)
class Prod6DiagnosticSnapshot:
    metrics: Tuple[Prod6MetricEvent, ...]
    audits: Tuple[Prod6AuditEvent, ...]
    spans: Tuple[Prod6SpanEvent, ...]
    dead_letters: Tuple[Prod6DeadLetterEvent, ...]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "metrics": [{"name": m.name.value, "value": m.value, "tags": dict(m.tags)} for m in self.metrics],
            "audits": [{"event_id": a.event_id, "action": a.action, "reason_codes": list(a.reason_codes)} for a in self.audits],
            "spans": [{"span_id": s.span_id, "name": s.name, "status": s.status, "safe_summary": dict(s.safe_summary)} for s in self.spans],
            "dead_letters": [{"record_id": d.record_id, "reason_codes": list(d.reason_codes), "safe_context": dict(d.safe_context)} for d in self.dead_letters],
        }

class Prod6ObservabilityCollector:
    def __init__(self):
        self.metrics = []
        self.audits = {}
        self.spans = {}
        self.dead_letters = {}

    def emit_metric(self, metric: Prod6MetricEvent):
        self.metrics.append(metric.validate())

    def emit_audit(self, audit: Prod6AuditEvent):
        audit.validate(); self.audits.setdefault(audit.event_id, audit)

    def emit_span(self, span: Prod6SpanEvent):
        span.validate(); self.spans.setdefault(span.span_id, span)

    def dead_letter(self, record: Prod6DeadLetterEvent):
        record.validate(); self.dead_letters.setdefault(record.record_id, record)

    def snapshot(self) -> Prod6DiagnosticSnapshot:
        return Prod6DiagnosticSnapshot(tuple(self.metrics), tuple(self.audits.values()), tuple(self.spans.values()), tuple(self.dead_letters.values()))


def build_default_prod6_observability_collector() -> Prod6ObservabilityCollector:
    return Prod6ObservabilityCollector()
