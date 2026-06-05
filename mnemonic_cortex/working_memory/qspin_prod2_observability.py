"""QSPIN-PROD-2 metadata-only observability records."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

class QSpinProd2MetricName(str, Enum):
    SHADOW_BUS_DISPATCH_ATTEMPTS="shadow_bus_dispatch_attempts"
    SHADOW_BUS_DISPATCH_BLOCKED="shadow_bus_dispatch_blocked"
    PAYLOAD_DRY_RUN_ATTEMPTS="payload_dry_run_attempts"
    PAYLOAD_DRY_RUN_BLOCKED="payload_dry_run_blocked"
    GUARDED_DISPATCH_SIMULATION_ATTEMPTS="guarded_dispatch_simulation_attempts"
    GUARDED_DISPATCH_SIMULATION_BLOCKED="guarded_dispatch_simulation_blocked"
    KILL_SWITCH_BLOCKS="kill_switch_blocks"
    COMMIT_GATE_DRY_RUN_BLOCKS="commit_gate_dry_run_blocks"
    ROLLBACK_EVIDENCE_BLOCKS="rollback_evidence_blocks"
    SOURCE_MATRIX_BLOCKS="source_matrix_blocks"
    RAW_PAYLOAD_REJECTION_COUNT="raw_payload_rejection_count"
    UNSAFE_WRITE_REJECTION_COUNT="unsafe_write_rejection_count"
@dataclass(frozen=True)
class QSpinProd2MetricRecord:
    name: QSpinProd2MetricName
    value: int=1
    tags: Mapping[str,str]=field(default_factory=dict)
    def validate(self):
        if not isinstance(self.name,QSpinProd2MetricName): raise ValueError("invalid metric name")
        if self.value < 0: raise ValueError("metric value must be non-negative")
        return self
@dataclass(frozen=True)
class QSpinProd2AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str,...]=()
    secret_free: bool=True
    raw_payload_free: bool=True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit event fields required")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe audit event")
        return self
@dataclass(frozen=True)
class QSpinProd2TraceRecord:
    trace_id: str
    safe_summary: Mapping[str, Any]
    contains_raw_payload: bool=False
    contains_secret: bool=False
    def validate(self):
        if not self.trace_id: raise ValueError("trace_id required")
        if self.contains_raw_payload or self.contains_secret: raise ValueError("unsafe trace record")
        return self
@dataclass(frozen=True)
class QSpinProd2DeadLetterRecord:
    record_id: str
    reason_codes: Tuple[str,...]
    safe_context: Mapping[str, Any]=field(default_factory=dict)
    contains_raw_payload: bool=False
    contains_secret: bool=False
    def validate(self):
        if not self.record_id or not self.reason_codes: raise ValueError("dead-letter requires ID and reason codes")
        if self.contains_raw_payload or self.contains_secret: raise ValueError("unsafe dead-letter")
        return self
@dataclass(frozen=True)
class QSpinProd2DiagnosticSnapshot:
    snapshot_id: str
    metrics: Tuple[QSpinProd2MetricRecord,...]
    audit_events: Tuple[QSpinProd2AuditEvent,...]
    traces: Tuple[QSpinProd2TraceRecord,...]
    dead_letters: Tuple[QSpinProd2DeadLetterRecord,...]
    def validate(self):
        for x in self.metrics: x.validate()
        for x in self.audit_events: x.validate()
        for x in self.traces: x.validate()
        for x in self.dead_letters: x.validate()
        return self
class QSpinProd2ObservabilityCollector:
    def __init__(self):
        self.metrics=[]; self.audit_events={}; self.traces={}; self.dead_letters={}
    def emit_metric(self, rec: QSpinProd2MetricRecord): self.metrics.append(rec.validate())
    def emit_audit(self, ev: QSpinProd2AuditEvent):
        ev.validate(); self.audit_events.setdefault(ev.event_id, ev)
    def emit_trace(self, tr: QSpinProd2TraceRecord):
        tr.validate(); self.traces.setdefault(tr.trace_id, tr)
    def dead_letter(self, rec: QSpinProd2DeadLetterRecord):
        rec.validate(); self.dead_letters.setdefault(rec.record_id, rec)
    def snapshot(self, snapshot_id="qspin_prod2_snapshot"):
        return QSpinProd2DiagnosticSnapshot(snapshot_id, tuple(self.metrics), tuple(self.audit_events.values()), tuple(self.traces.values()), tuple(self.dead_letters.values())).validate()
def build_default_qspin_prod2_observability_collector(): return QSpinProd2ObservabilityCollector()
