"""QSPIN-PROD-4 metadata-only observability records."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple

class QSpinProd4MetricName(str, Enum):
    SYNTHETIC_PAYLOAD_EXECUTION_ATTEMPTS="synthetic_payload_execution_attempts"
    SYNTHETIC_PAYLOAD_EXECUTION_BLOCKED="synthetic_payload_execution_blocked"
    SYNTHETIC_SANDBOX_OPERATION_ATTEMPTS="synthetic_sandbox_operation_attempts"
    SYNTHETIC_SANDBOX_OPERATION_BLOCKED="synthetic_sandbox_operation_blocked"
    EXPANDED_COMMIT_GATE_DRY_RUN_ATTEMPTS="expanded_commit_gate_dry_run_attempts"
    EXPANDED_COMMIT_GATE_DRY_RUN_BLOCKED="expanded_commit_gate_dry_run_blocked"
    RUNTIME_SAFETY_REGRESSION_CASES_RUN="runtime_safety_regression_cases_run"
    RUNTIME_SAFETY_REGRESSION_CASES_FAILED="runtime_safety_regression_cases_failed"
    QH_WRITE_REJECTION_COUNT="qh_write_rejection_count"
    SHARED_SLOT_WRITE_REJECTION_COUNT="shared_slot_write_rejection_count"
    EXTERNAL_MEMORY_WRITE_REJECTION_COUNT="external_memory_write_rejection_count"
    RAW_PAYLOAD_REJECTION_COUNT="raw_payload_rejection_count"
    UNSAFE_SHAPE_REJECTION_COUNT="unsafe_shape_rejection_count"
    BUDGET_EXCEEDANCE_REJECTION_COUNT="budget_exceedance_rejection_count"
    PRODUCTION_ACTIVATION_REJECTION_COUNT="production_activation_rejection_count"
    NO_MUTATION_ASSERTIONS_PASSED="no_mutation_assertions_passed"

@dataclass(frozen=True)
class QSpinProd4MetricRecord:
    name: QSpinProd4MetricName
    value: int=1
    tags: Mapping[str,str]=field(default_factory=dict)
    def validate(self):
        if not isinstance(self.name,QSpinProd4MetricName): raise ValueError("invalid metric name")
        if self.value < 0: raise ValueError("metric value must be non-negative")
        return self

@dataclass(frozen=True)
class QSpinProd4AuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str,...]=()
    secret_free: bool=True
    raw_payload_free: bool=True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit fields required")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe audit event")
        return self

@dataclass(frozen=True)
class QSpinProd4TraceRecord:
    trace_id: str
    safe_summary: Mapping[str, Any]
    contains_raw_payload: bool=False
    contains_secret: bool=False
    def validate(self):
        if not self.trace_id: raise ValueError("trace_id required")
        if self.contains_raw_payload or self.contains_secret: raise ValueError("unsafe trace")
        return self

@dataclass(frozen=True)
class QSpinProd4DeadLetterRecord:
    record_id: str
    reason_codes: Tuple[str,...]
    safe_context: Mapping[str,Any]=field(default_factory=dict)
    contains_raw_payload: bool=False
    contains_secret: bool=False
    def validate(self):
        if not self.record_id or not self.reason_codes: raise ValueError("dead-letter fields required")
        if self.contains_raw_payload or self.contains_secret: raise ValueError("unsafe dead-letter")
        return self

@dataclass(frozen=True)
class QSpinProd4DiagnosticSnapshot:
    snapshot_id: str
    metrics: Tuple[QSpinProd4MetricRecord,...]
    audit_events: Tuple[QSpinProd4AuditEvent,...]
    traces: Tuple[QSpinProd4TraceRecord,...]
    dead_letters: Tuple[QSpinProd4DeadLetterRecord,...]
    def validate(self):
        for x in self.metrics: x.validate()
        for x in self.audit_events: x.validate()
        for x in self.traces: x.validate()
        for x in self.dead_letters: x.validate()
        return self

class QSpinProd4ObservabilityCollector:
    def __init__(self):
        self.metrics=[]; self.audit_events={}; self.traces={}; self.dead_letters={}
    def emit_metric(self, rec: QSpinProd4MetricRecord): self.metrics.append(rec.validate())
    def emit_audit(self, ev: QSpinProd4AuditEvent): ev.validate(); self.audit_events.setdefault(ev.event_id,ev)
    def emit_trace(self, tr: QSpinProd4TraceRecord): tr.validate(); self.traces.setdefault(tr.trace_id,tr)
    def dead_letter(self, rec: QSpinProd4DeadLetterRecord): rec.validate(); self.dead_letters.setdefault(rec.record_id,rec)
    def snapshot(self, snapshot_id="qspin_prod4_snapshot"):
        return QSpinProd4DiagnosticSnapshot(snapshot_id,tuple(self.metrics),tuple(self.audit_events.values()),tuple(self.traces.values()),tuple(self.dead_letters.values())).validate()

def build_default_qspin_prod4_observability_collector(): return QSpinProd4ObservabilityCollector()
