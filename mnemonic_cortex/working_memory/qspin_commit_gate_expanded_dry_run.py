"""QSPIN-PROD-4 expanded commit-gate approval dry-run."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple, FrozenSet

class QSpinCommitGateExpandedMode(str, Enum):
    DISABLED="disabled"
    DRY_RUN_ONLY="dry_run_only"

class QSpinCommitGateExpandedStatus(str, Enum):
    APPROVED_DRY_RUN="approved_dry_run"
    BLOCKED="blocked"

class QSpinCommitGateExpandedBlockReason(str, Enum):
    MODE_DISABLED="mode_disabled"
    MISSING_EVIDENCE="missing_evidence"
    UNSAFE_RUNTIME_FLAG="unsafe_runtime_flag"
    WRITE_INTENT="write_intent"
    RAW_TRACE_INTENT="raw_trace_intent"
    PRODUCTION_ACTIVATION="production_activation"
    REAL_PAYLOAD_TRANSFER="real_payload_transfer"
    COMMIT_EXECUTION="commit_execution"

class QSpinCommitGateEvidenceKind(str, Enum):
    QD6A_MATRIX="qd6a_matrix"
    QSPIN_PRESERVATION="qspin_preservation"
    PROD_PRESERVATION="prod_preservation"
    KILL_SWITCH_ENABLED="kill_switch_enabled"
    ROLLBACK_DRY_RUN="rollback_dry_run"
    SHADOW_ACTIVATION="shadow_activation"
    SHADOW_BUS="shadow_bus"
    PAYLOAD_DRY_RUN="payload_dry_run"
    GUARDED_DISPATCH="guarded_dispatch"
    PAYLOAD_ROUNDTRIP="payload_roundtrip"
    PERMISSION_DRY_RUN="permission_dry_run"
    ACTIVE_DRY_RUN_EXECUTOR="active_dry_run_executor"
    SYNTHETIC_PAYLOAD_HARNESS="synthetic_payload_harness"
    SYNTHETIC_SANDBOX="synthetic_sandbox"
    TRACE_SAFE_LOGGING="trace_safe_logging"
    OBSERVABILITY="observability"

@dataclass(frozen=True)
class QSpinCommitGateEvidenceRecord:
    kind: QSpinCommitGateEvidenceKind
    evidence_id: str
    present: bool=True
    def validate(self):
        if not isinstance(self.kind,QSpinCommitGateEvidenceKind): raise ValueError("invalid evidence kind")
        if not self.evidence_id: raise ValueError("evidence_id required")
        return self

@dataclass(frozen=True)
class QSpinCommitGateExpandedPolicy:
    mode: QSpinCommitGateExpandedMode=QSpinCommitGateExpandedMode.DRY_RUN_ONLY
    required_evidence: FrozenSet[QSpinCommitGateEvidenceKind]=frozenset(QSpinCommitGateEvidenceKind)
    def validate(self):
        if self.mode is QSpinCommitGateExpandedMode.DISABLED: raise ValueError("expanded commit gate disabled")
        if set(self.required_evidence) != set(QSpinCommitGateEvidenceKind): raise ValueError("all evidence kinds required")
        return self

@dataclass(frozen=True)
class QSpinCommitGateExpandedRequest:
    request_id: str
    evidence: Tuple[QSpinCommitGateEvidenceRecord,...]
    unsafe_runtime_flags: Tuple[str,...]=()
    write_intent: bool=False
    raw_trace_intent: bool=False
    production_activation_requested: bool=False
    real_payload_transfer_requested: bool=False
    commit_execution_requested: bool=False
    def validate(self):
        if not self.request_id: raise ValueError("request_id required")
        for e in self.evidence: e.validate()
        return self

@dataclass(frozen=True)
class QSpinCommitGateExpandedDecision:
    status: QSpinCommitGateExpandedStatus
    approved: bool
    missing_evidence: Tuple[QSpinCommitGateEvidenceKind,...]=()
    block_reasons: Tuple[QSpinCommitGateExpandedBlockReason,...]=()
    def validate(self):
        if self.status is QSpinCommitGateExpandedStatus.APPROVED_DRY_RUN and (not self.approved or self.missing_evidence or self.block_reasons): raise ValueError("bad expanded gate approval")
        if self.status is QSpinCommitGateExpandedStatus.BLOCKED and (self.approved or not self.block_reasons): raise ValueError("bad expanded gate block")
        return self

@dataclass(frozen=True)
class QSpinCommitGateExpandedTrace:
    request_id: str
    status: QSpinCommitGateExpandedStatus
    safe_summary: Mapping[str, Any]
    def to_dict(self): return {"request_id":self.request_id,"status":self.status.value,"safe_summary":dict(self.safe_summary)}

@dataclass(frozen=True)
class QSpinCommitGateExpandedResult:
    request: QSpinCommitGateExpandedRequest
    decision: QSpinCommitGateExpandedDecision
    trace: QSpinCommitGateExpandedTrace
    commit_executed: bool=False
    runtime_activated: bool=False
    def validate(self):
        self.decision.validate()
        if self.commit_executed or self.runtime_activated: raise ValueError("expanded commit gate dry-run must not commit or activate")
        return self

class QSpinCommitGateExpandedApprovalDryRun:
    def __init__(self, policy: QSpinCommitGateExpandedPolicy|None=None):
        self.policy=(policy or build_default_qspin_commit_gate_expanded_policy()).validate()
    def inspect(self, request: QSpinCommitGateExpandedRequest) -> QSpinCommitGateExpandedResult:
        request.validate()
        present={e.kind for e in request.evidence if e.present}
        missing=tuple(sorted(set(self.policy.required_evidence)-present,key=lambda x:x.value))
        reasons=[]
        if missing: reasons.append(QSpinCommitGateExpandedBlockReason.MISSING_EVIDENCE)
        if request.unsafe_runtime_flags: reasons.append(QSpinCommitGateExpandedBlockReason.UNSAFE_RUNTIME_FLAG)
        if request.write_intent: reasons.append(QSpinCommitGateExpandedBlockReason.WRITE_INTENT)
        if request.raw_trace_intent: reasons.append(QSpinCommitGateExpandedBlockReason.RAW_TRACE_INTENT)
        if request.production_activation_requested: reasons.append(QSpinCommitGateExpandedBlockReason.PRODUCTION_ACTIVATION)
        if request.real_payload_transfer_requested: reasons.append(QSpinCommitGateExpandedBlockReason.REAL_PAYLOAD_TRANSFER)
        if request.commit_execution_requested: reasons.append(QSpinCommitGateExpandedBlockReason.COMMIT_EXECUTION)
        if reasons:
            decision=QSpinCommitGateExpandedDecision(QSpinCommitGateExpandedStatus.BLOCKED,False,missing,tuple(dict.fromkeys(reasons))).validate()
        else:
            decision=QSpinCommitGateExpandedDecision(QSpinCommitGateExpandedStatus.APPROVED_DRY_RUN,True).validate()
        trace=QSpinCommitGateExpandedTrace(request.request_id,decision.status,{"missing_evidence":[m.value for m in missing],"block_reasons":[r.value for r in decision.block_reasons],"evidence_count":len(present)})
        return QSpinCommitGateExpandedResult(request,decision,trace).validate()

def build_default_qspin_commit_gate_expanded_policy(): return QSpinCommitGateExpandedPolicy().validate()
