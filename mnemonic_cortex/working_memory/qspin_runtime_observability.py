"""QSPIN-PROD-5 synthetic runtime observability primitives.

Safety posture:
- local-only
- no networking
- no production writes
- redaction-safe serialization
- deterministic summaries
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
import json
import re
import time

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9]{12,}"),
    re.compile(r"(?i)-----BEGIN\s+(RSA|OPENSSH|PRIVATE)\s+KEY-----"),
]

REDACTED = "<REDACTED>"


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    unit: str = "count"
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: round(time.time(), 6))


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    stage: str
    status: str
    reason_code: str
    detail: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: round(time.time(), 6))


@dataclass(frozen=True)
class SpanEvent:
    span_name: str
    started_at: float
    ended_at: float
    status: str
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.ended_at - self.started_at) * 1000.0, 3)


def redact_text(value: str) -> str:
    out = value
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(k): redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    return value


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def no_secret_leakage_check(value: Any) -> bool:
    text = json.dumps(to_jsonable(value), sort_keys=True, default=str)
    return not any(pattern.search(text) for pattern in SECRET_PATTERNS)


class ObservabilityEmitter:
    """In-memory structured event emitter for synthetic QSPIN runs."""

    def __init__(self, *, stage: str = "QSPIN-PROD-5-QD6A", lineage: Optional[Dict[str, str]] = None):
        self.stage = stage
        self.lineage = lineage or {}
        self.metrics: List[MetricEvent] = []
        self.audits: List[AuditEvent] = []
        self.spans: List[SpanEvent] = []
        self.counters: Dict[str, int] = {
            "audit_events": 0,
            "metric_events": 0,
            "span_events": 0,
            "fail_closed": 0,
            "skips": 0,
            "canaries": 0,
            "remediations": 0,
        }

    def emit_metric(self, name: str, value: float, *, unit: str = "count", tags: Optional[Dict[str, str]] = None) -> MetricEvent:
        event = MetricEvent(name=name, value=float(value), unit=unit, tags=redact_value(tags or {}))
        self.metrics.append(event)
        self.counters["metric_events"] += 1
        return event

    def emit_audit(self, event_type: str, status: str, reason_code: str, *, detail: Optional[Dict[str, Any]] = None) -> AuditEvent:
        clean_detail = redact_value(detail or {})
        event = AuditEvent(
            event_type=event_type,
            stage=self.stage,
            status=status,
            reason_code=reason_code,
            detail=clean_detail,
            lineage=self.lineage,
        )
        self.audits.append(event)
        self.counters["audit_events"] += 1
        if status == "FAIL_CLOSED":
            self.counters["fail_closed"] += 1
        if status == "SKIP":
            self.counters["skips"] += 1
        return event

    def emit_span(self, span_name: str, started_at: float, ended_at: float, status: str, *, tags: Optional[Dict[str, str]] = None) -> SpanEvent:
        event = SpanEvent(span_name=span_name, started_at=started_at, ended_at=ended_at, status=status, tags=redact_value(tags or {}))
        self.spans.append(event)
        self.counters["span_events"] += 1
        self.emit_metric(f"latency.{span_name}", event.duration_ms, unit="ms", tags=tags)
        return event

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + int(amount)

    def summary(self) -> Dict[str, Any]:
        payload = {
            "stage": self.stage,
            "lineage": self.lineage,
            "counters": dict(sorted(self.counters.items())),
            "metrics": [to_jsonable(v) for v in self.metrics],
            "audits": [to_jsonable(v) for v in self.audits],
            "spans": [to_jsonable(v) for v in self.spans],
        }
        return redact_value(payload)

    def to_json(self) -> str:
        payload = self.summary()
        if not no_secret_leakage_check(payload):
            raise ValueError("observability payload failed no_secret_leakage_check")
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
