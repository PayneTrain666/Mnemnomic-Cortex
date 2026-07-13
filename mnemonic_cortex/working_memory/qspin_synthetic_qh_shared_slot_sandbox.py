"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-4 synthetic QH/shared-slot sandbox.

The sandbox is local metadata only. It is isolated from real QD6A shared-slot,
QH, and external-memory stores. Real writes are rejected. Sandbox-local writes
are allowed only as synthetic metadata records when policy permits them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib, json

class QSpinSyntheticSandboxMode(str, Enum):
    DISABLED="disabled"
    SANDBOX_ONLY="sandbox_only"

class QSpinSyntheticSandboxStatus(str, Enum):
    SIMULATED="simulated"
    BLOCKED="blocked"
    CLEANED="cleaned"

class QSpinSyntheticSandboxBlockReason(str, Enum):
    MODE_DISABLED="mode_disabled"
    UNKNOWN_SCOPE="unknown_scope"
    UNKNOWN_OPERATION="unknown_operation"
    REAL_WRITE_REQUESTED="real_write_requested"
    REAL_READ_REQUESTED="real_read_requested"
    RAW_PAYLOAD_PRESENT="raw_payload_present"
    RAW_TENSOR_PRESENT="raw_tensor_present"
    INTERFERENCE_CHECK_MISSING="interference_check_missing"
    COMMIT_GATE_REVIEW_MISSING="commit_gate_review_missing"
    PERMISSION_METADATA_MISSING="permission_metadata_missing"

class QSpinSyntheticSandboxScope(str, Enum):
    SHARED_SLOT="shared_slot"
    QH="qh"
    EXTERNAL_MEMORY="external_memory"

class QSpinSyntheticSandboxOperation(str, Enum):
    READ_METADATA="read_metadata"
    WRITE_SANDBOX="write_sandbox"
    WRITE_REAL="write_real"
    READ_REAL="read_real"
    CLEANUP="cleanup"

@dataclass(frozen=True)
class QSpinSyntheticSharedSlotRecord:
    record_id: str
    slot_id: str
    metadata: Mapping[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.record_id or not self.slot_id: raise ValueError("shared-slot record fields required")
        return self

@dataclass(frozen=True)
class QSpinSyntheticQHRecord:
    record_id: str
    qh_cell_id: str
    interference_score: float = 0.0
    metadata: Mapping[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.record_id or not self.qh_cell_id: raise ValueError("QH record fields required")
        if self.interference_score < 0: raise ValueError("invalid interference score")
        return self

@dataclass(frozen=True)
class QSpinSyntheticSandboxPermissionPolicy:
    mode: QSpinSyntheticSandboxMode=QSpinSyntheticSandboxMode.SANDBOX_ONLY
    allow_sandbox_writes: bool=True
    allow_real_reads: bool=False
    allow_real_writes: bool=False
    require_interference_check: bool=True
    require_commit_gate_review: bool=True
    require_permission_metadata: bool=True
    def validate(self):
        if self.mode is QSpinSyntheticSandboxMode.DISABLED: raise ValueError("sandbox disabled")
        if self.allow_real_reads or self.allow_real_writes: raise ValueError("real reads/writes forbidden")
        return self

@dataclass(frozen=True)
class QSpinSyntheticSandboxRequest:
    request_id: str
    scope: QSpinSyntheticSandboxScope
    operation: QSpinSyntheticSandboxOperation
    target_id: str
    interference_check_present: bool
    commit_gate_review_present: bool
    permission_metadata_present: bool
    raw_payload: Any=None
    raw_tensor: Any=None
    metadata: Mapping[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.request_id or not self.target_id: raise ValueError("sandbox request fields required")
        if not isinstance(self.scope,QSpinSyntheticSandboxScope): raise ValueError("unknown sandbox scope")
        if not isinstance(self.operation,QSpinSyntheticSandboxOperation): raise ValueError("unknown sandbox operation")
        if self.raw_payload is not None: raise ValueError("raw payload forbidden")
        if self.raw_tensor is not None: raise ValueError("raw tensor forbidden")
        return self
    def record_id(self):
        data={"scope":self.scope.value,"op":self.operation.value,"target":self.target_id,"metadata_keys":sorted(str(k) for k in self.metadata.keys())}
        return "sandbox_" + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]

@dataclass(frozen=True)
class QSpinSyntheticSandboxDecision:
    status: QSpinSyntheticSandboxStatus
    allowed: bool
    block_reasons: Tuple[QSpinSyntheticSandboxBlockReason,...]=()
    def validate(self):
        if self.status in {QSpinSyntheticSandboxStatus.SIMULATED,QSpinSyntheticSandboxStatus.CLEANED} and (not self.allowed or self.block_reasons): raise ValueError("bad sandbox allowed decision")
        if self.status is QSpinSyntheticSandboxStatus.BLOCKED and (self.allowed or not self.block_reasons): raise ValueError("bad sandbox block decision")
        return self

@dataclass(frozen=True)
class QSpinSyntheticSandboxTrace:
    request_id: str
    status: QSpinSyntheticSandboxStatus
    safe_summary: Mapping[str, Any]
    def to_dict(self): return {"request_id":self.request_id,"status":self.status.value,"safe_summary":dict(self.safe_summary)}

@dataclass(frozen=True)
class QSpinSyntheticSandboxAuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str,...]=()
    secret_free: bool=True
    raw_payload_free: bool=True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit event fields required")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe sandbox audit")
        return self

@dataclass(frozen=True)
class QSpinSyntheticSandboxResult:
    request: QSpinSyntheticSandboxRequest
    decision: QSpinSyntheticSandboxDecision
    trace: QSpinSyntheticSandboxTrace
    audit_event: QSpinSyntheticSandboxAuditEvent
    touched_real_store: bool=False
    wrote_real_store: bool=False
    def validate(self):
        self.decision.validate(); self.audit_event.validate()
        if self.touched_real_store or self.wrote_real_store: raise ValueError("sandbox must not touch real stores")
        return self

class QSpinSyntheticQHSharedSlotSandbox:
    def __init__(self, policy: Optional[QSpinSyntheticSandboxPermissionPolicy]=None):
        self.policy=(policy or build_default_qspin_synthetic_sandbox_policy()).validate()
        self.shared_slot_records: Dict[str,QSpinSyntheticSharedSlotRecord]={}
        self.qh_records: Dict[str,QSpinSyntheticQHRecord]={}
        self.cleaned=False
    def operate(self, request: QSpinSyntheticSandboxRequest) -> QSpinSyntheticSandboxResult:
        reasons=[]
        try: request.validate()
        except ValueError as exc:
            msg=str(exc)
            if "raw payload" in msg: reasons.append(QSpinSyntheticSandboxBlockReason.RAW_PAYLOAD_PRESENT)
            elif "raw tensor" in msg: reasons.append(QSpinSyntheticSandboxBlockReason.RAW_TENSOR_PRESENT)
            elif "scope" in msg: reasons.append(QSpinSyntheticSandboxBlockReason.UNKNOWN_SCOPE)
            else: reasons.append(QSpinSyntheticSandboxBlockReason.UNKNOWN_OPERATION)
        if request.operation is QSpinSyntheticSandboxOperation.WRITE_REAL: reasons.append(QSpinSyntheticSandboxBlockReason.REAL_WRITE_REQUESTED)
        if request.operation is QSpinSyntheticSandboxOperation.READ_REAL: reasons.append(QSpinSyntheticSandboxBlockReason.REAL_READ_REQUESTED)
        if not request.interference_check_present: reasons.append(QSpinSyntheticSandboxBlockReason.INTERFERENCE_CHECK_MISSING)
        if not request.commit_gate_review_present: reasons.append(QSpinSyntheticSandboxBlockReason.COMMIT_GATE_REVIEW_MISSING)
        if not request.permission_metadata_present: reasons.append(QSpinSyntheticSandboxBlockReason.PERMISSION_METADATA_MISSING)
        if reasons:
            decision=QSpinSyntheticSandboxDecision(QSpinSyntheticSandboxStatus.BLOCKED,False,tuple(dict.fromkeys(reasons))).validate()
        elif request.operation is QSpinSyntheticSandboxOperation.CLEANUP:
            self.shared_slot_records.clear(); self.qh_records.clear(); self.cleaned=True
            decision=QSpinSyntheticSandboxDecision(QSpinSyntheticSandboxStatus.CLEANED,True).validate()
        else:
            if request.operation is QSpinSyntheticSandboxOperation.WRITE_SANDBOX and self.policy.allow_sandbox_writes:
                rid=request.record_id()
                if request.scope is QSpinSyntheticSandboxScope.SHARED_SLOT:
                    self.shared_slot_records.setdefault(rid,QSpinSyntheticSharedSlotRecord(rid,request.target_id,request.metadata).validate())
                if request.scope is QSpinSyntheticSandboxScope.QH:
                    self.qh_records.setdefault(rid,QSpinSyntheticQHRecord(rid,request.target_id,0.0,request.metadata).validate())
            decision=QSpinSyntheticSandboxDecision(QSpinSyntheticSandboxStatus.SIMULATED,True).validate()
        summary={"scope":request.scope.value if isinstance(request.scope,QSpinSyntheticSandboxScope) else str(request.scope),"operation":request.operation.value if isinstance(request.operation,QSpinSyntheticSandboxOperation) else str(request.operation),"target_id":request.target_id,"record_id":request.record_id() if isinstance(request.scope,QSpinSyntheticSandboxScope) and isinstance(request.operation,QSpinSyntheticSandboxOperation) else "invalid","shared_slot_record_count":len(self.shared_slot_records),"qh_record_count":len(self.qh_records),"block_count":len(decision.block_reasons)}
        trace=QSpinSyntheticSandboxTrace(request.request_id,decision.status,summary)
        audit=QSpinSyntheticSandboxAuditEvent("audit_"+request.request_id,"sandbox_operation",tuple(r.value for r in decision.block_reasons)).validate()
        return QSpinSyntheticSandboxResult(request,decision,trace,audit).validate()
    def cleanup(self):
        req=QSpinSyntheticSandboxRequest("cleanup",QSpinSyntheticSandboxScope.SHARED_SLOT,QSpinSyntheticSandboxOperation.CLEANUP,"sandbox",True,True,True)
        return self.operate(req)

def build_default_qspin_synthetic_sandbox_policy(): return QSpinSyntheticSandboxPermissionPolicy().validate()
def build_default_qspin_synthetic_qh_shared_slot_sandbox(): return QSpinSyntheticQHSharedSlotSandbox()
