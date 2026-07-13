"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-3 metadata-only observability records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple


class QSpinProd3MetricName(str, Enum):
    ACTIVE_DRY_RUN_EXECUTION_ATTEMPTS = "active_dry_run_execution_attempts"
    ACTIVE_DRY_RUN_EXECUTION_BLOCKED = "active_dry_run_execution_blocked"
    COMMIT_GATE_APPROVAL_SIM_ATTEMPTS = "commit_gate_approval_sim_attempts"
    COMMIT_GATE_APPROVAL_SIM_BLOCKED = "commit_gate_approval_sim_blocked"
    PAYLOAD_ROUNDTRIP_STUB_ATTEMPTS = "payload_roundtrip_stub_attempts"
    PAYLOAD_ROUNDTRIP_STUB_BLOCKED = "payload_roundtrip_stub_blocked"
    PERMISSION_DRY_RUN_ATTEMPTS = "permission_dry_run_attempts"
    PERMISSION_DRY_RUN_BLOCKED = "permission_dry_run_blocked"
    QH_WRITE_REJECTION_COUNT = "qh_write_rejection_count"
    SHARED_SLOT_WRITE_REJECTION_COUNT = "shared_slot_write_rejection_count"
    EXTERNAL_MEMORY_WRITE_REJECTION_COUNT = "external_memory_write_rejection_count"
    RAW_PAYLOAD_REJECTION_COUNT = "raw_payload_rejection_count"
    UNSAFE_RUNTIME_FLAG_REJECTION_COUNT = "unsafe_runtime_flag_rejection_count"
    PRODUCTION_ACTIVATION_REJECTION_COUNT = "production_activation_rejection_count"


@dataclass(frozen=True)
class QSpinProd3MetricRecord:
    name: QSpinProd3MetricName
    value: int = 1
    tags: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> "QSpinProd3MetricRecord":
        if not isinstance(self.name, QSpinProd3MetricName):
            raise ValueError("invalid metric name")
        if self.value < 0:
            raise ValueError("metric value must be non-negative")
        return self


@dataclass(frozen=True)
class QSpinProd3AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    secret_free: bool = True
    raw_payload_free: bool = True

    def validate(self) -> "QSpinProd3AuditEvent":
        if not self.event_id or not self.action:
            raise ValueError("audit event fields required")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("unsafe audit event")
        return self


@dataclass(frozen=True)
class QSpinProd3TraceRecord:
    trace_id: str
    safe_summary: Mapping[str, Any]
    contains_raw_payload: bool = False
    contains_secret: bool = False

    def validate(self) -> "QSpinProd3TraceRecord":
        if not self.trace_id:
            raise ValueError("trace_id required")
        if self.contains_raw_payload or self.contains_secret:
            raise ValueError("unsafe trace record")
        return self


@dataclass(frozen=True)
class QSpinProd3DeadLetterRecord:
    record_id: str
    reason_codes: Tuple[str, ...]
    safe_context: Mapping[str, Any] = field(default_factory=dict)
    contains_raw_payload: bool = False
    contains_secret: bool = False

    def validate(self) -> "QSpinProd3DeadLetterRecord":
        if not self.record_id or not self.reason_codes:
            raise ValueError("dead-letter requires ID and reason codes")
        if self.contains_raw_payload or self.contains_secret:
            raise ValueError("unsafe dead-letter")
        return self


@dataclass(frozen=True)
class QSpinProd3DiagnosticSnapshot:
    snapshot_id: str
    metrics: Tuple[QSpinProd3MetricRecord, ...]
    audit_events: Tuple[QSpinProd3AuditEvent, ...]
    traces: Tuple[QSpinProd3TraceRecord, ...]
    dead_letters: Tuple[QSpinProd3DeadLetterRecord, ...]

    def validate(self) -> "QSpinProd3DiagnosticSnapshot":
        for rec in self.metrics:
            rec.validate()
        for ev in self.audit_events:
            ev.validate()
        for tr in self.traces:
            tr.validate()
        for dl in self.dead_letters:
            dl.validate()
        return self


class QSpinProd3ObservabilityCollector:
    def __init__(self):
        self.metrics = []
        self.audit_events = {}
        self.traces = {}
        self.dead_letters = {}

    def emit_metric(self, rec: QSpinProd3MetricRecord):
        self.metrics.append(rec.validate())

    def emit_audit(self, ev: QSpinProd3AuditEvent):
        ev.validate()
        self.audit_events.setdefault(ev.event_id, ev)

    def emit_trace(self, tr: QSpinProd3TraceRecord):
        tr.validate()
        self.traces.setdefault(tr.trace_id, tr)

    def dead_letter(self, rec: QSpinProd3DeadLetterRecord):
        rec.validate()
        self.dead_letters.setdefault(rec.record_id, rec)

    def snapshot(self, snapshot_id: str = "qspin_prod3_snapshot") -> QSpinProd3DiagnosticSnapshot:
        return QSpinProd3DiagnosticSnapshot(
            snapshot_id,
            tuple(self.metrics),
            tuple(self.audit_events.values()),
            tuple(self.traces.values()),
            tuple(self.dead_letters.values()),
        ).validate()


def build_default_qspin_prod3_observability_collector() -> QSpinProd3ObservabilityCollector:
    return QSpinProd3ObservabilityCollector()
