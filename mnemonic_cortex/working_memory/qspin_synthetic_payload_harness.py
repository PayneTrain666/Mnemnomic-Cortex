"""QSPIN-PROD-4 guarded synthetic payload execution harness.

Synthetic metadata only. This module never uses real user payloads, never stores
raw tensors, never transfers payloads, never writes shared slots/external
memory/QH storage, never executes commits, and never activates production.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib, json

class QSpinSyntheticPayloadMode(str, Enum):
    DISABLED = "disabled"
    SYNTHETIC_ONLY = "synthetic_only"

class QSpinSyntheticPayloadStatus(str, Enum):
    EXECUTED_SYNTHETIC_ONLY = "executed_synthetic_only"
    BLOCKED = "blocked"

class QSpinSyntheticPayloadBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    REAL_PAYLOAD_PRESENT = "real_payload_present"
    RAW_TENSOR_PRESENT = "raw_tensor_present"
    UNSAFE_SHAPE = "unsafe_shape"
    BUDGET_EXCEEDED = "budget_exceeded"
    NORM_BAND_INVALID = "norm_band_invalid"
    MISSING_ROUNDTRIP_APPROVAL = "missing_roundtrip_approval"
    MISSING_PERMISSION_APPROVAL = "missing_permission_approval"
    MISSING_EXECUTOR_APPROVAL = "missing_executor_approval"
    MISSING_PAYLOAD_DRY_RUN_APPROVAL = "missing_payload_dry_run_approval"
    MISSING_DISPATCH_APPROVAL = "missing_dispatch_approval"
    MISSING_KILL_SWITCH_APPROVAL = "missing_kill_switch_approval"
    MISSING_ROLLBACK_EVIDENCE = "missing_rollback_evidence"
    MISSING_COMMIT_GATE_APPROVAL = "missing_commit_gate_approval"
    WRITE_REQUESTED = "write_requested"
    COMMIT_REQUESTED = "commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"

class QSpinSyntheticPayloadKind(str, Enum):
    DENSE = "dense"
    CHRR = "chrr"
    QH = "qh"
    PHASE_LINK = "phase_link"
    DEPTH_LINK = "depth_link"
    MAP_BRIDGE = "map_bridge"

class QSpinSyntheticPayloadLifecycle(str, Enum):
    PLANNED = "planned"
    GENERATED_METADATA = "generated_metadata"
    ROUNDTRIP_STUBBED = "roundtrip_stubbed"
    PERMISSION_CHECKED = "permission_checked"
    EXECUTOR_DRY_RUN_PASSED = "executor_dry_run_passed"
    DISCARDED = "discarded"

@dataclass(frozen=True)
class QSpinSyntheticPayloadShape:
    dims: Tuple[int, ...]
    dtype: str = "float32"
    max_rank: int = 4
    max_elements: int = 1_000_000
    def element_count(self) -> int:
        n=1
        for d in self.dims: n*=d
        return n
    def validate(self):
        if not self.dims or len(self.dims) > self.max_rank:
            raise ValueError("unsafe synthetic payload shape rank")
        if any((not isinstance(d,int)) or d <= 0 for d in self.dims):
            raise ValueError("synthetic payload dims must be positive integers")
        if self.element_count() > self.max_elements:
            raise ValueError("synthetic payload element budget exceeded")
        if not self.dtype or any(ch in self.dtype for ch in "\n\r\t"):
            raise ValueError("unsafe dtype")
        return self
    def to_dict(self): return {"dims":list(self.dims),"dtype":self.dtype,"element_count":self.element_count()}

@dataclass(frozen=True)
class QSpinSyntheticPayloadBudget:
    declared_bytes: int
    max_bytes: int = 16_777_216
    bandwidth_units: int = 1
    max_bandwidth_units: int = 16
    def validate(self):
        if self.declared_bytes < 0 or self.max_bytes <= 0 or self.declared_bytes > self.max_bytes:
            raise ValueError("synthetic payload byte budget exceeded")
        if self.bandwidth_units < 0 or self.bandwidth_units > self.max_bandwidth_units:
            raise ValueError("synthetic payload bandwidth budget exceeded")
        return self
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QSpinSyntheticPayloadEnvelope:
    payload_kind: QSpinSyntheticPayloadKind
    source_id: str
    target_id: str
    shape: QSpinSyntheticPayloadShape
    budget: QSpinSyntheticPayloadBudget
    norm_band: Tuple[float, float] = (0.0, 1.0)
    lifecycle: Tuple[QSpinSyntheticPayloadLifecycle, ...] = (QSpinSyntheticPayloadLifecycle.PLANNED,)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def validate(self):
        if not isinstance(self.payload_kind,QSpinSyntheticPayloadKind): raise ValueError("invalid synthetic payload kind")
        if not self.source_id or not self.target_id: raise ValueError("source_id and target_id required")
        self.shape.validate(); self.budget.validate()
        lo,hi=self.norm_band
        if lo < 0 or hi < lo or hi > 1e6: raise ValueError("norm band invalid")
        for step in self.lifecycle:
            if not isinstance(step,QSpinSyntheticPayloadLifecycle): raise ValueError("invalid lifecycle step")
        return self
    def synthetic_payload_id(self) -> str:
        data={"kind":self.payload_kind.value,"source":self.source_id,"target":self.target_id,"shape":self.shape.to_dict(),"budget":self.budget.to_dict(),"norm":list(self.norm_band)}
        return "syn_" + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]
    def to_safe_dict(self):
        return {"payload_kind":self.payload_kind.value,"source_id":self.source_id,"target_id":self.target_id,"shape":self.shape.to_dict(),"budget":self.budget.to_dict(),"norm_band":list(self.norm_band),"lifecycle":[s.value for s in self.lifecycle],"synthetic_payload_id":self.synthetic_payload_id(),"metadata_keys":sorted(str(k) for k in self.metadata.keys())}

@dataclass(frozen=True)
class QSpinSyntheticPayloadGeneratorPolicy:
    mode: QSpinSyntheticPayloadMode = QSpinSyntheticPayloadMode.SYNTHETIC_ONLY
    allow_real_payloads: bool = False
    allow_raw_tensors: bool = False
    allow_payload_transfer: bool = False
    allow_writes: bool = False
    allow_commit_execution: bool = False
    allow_production_activation: bool = False
    def validate(self):
        if self.mode is QSpinSyntheticPayloadMode.DISABLED: raise ValueError("synthetic payload harness disabled")
        if self.allow_real_payloads or self.allow_raw_tensors or self.allow_payload_transfer or self.allow_writes or self.allow_commit_execution or self.allow_production_activation:
            raise ValueError("unsafe synthetic payload policy")
        return self

@dataclass(frozen=True)
class QSpinSyntheticPayloadExecutionRequest:
    request_id: str
    envelope: QSpinSyntheticPayloadEnvelope
    roundtrip_approved: bool
    permission_approved: bool
    executor_approved: bool
    payload_dry_run_approved: bool
    dispatch_approved: bool
    kill_switch_allows: bool
    rollback_evidence_present: bool
    commit_gate_approved: bool
    real_payload: Any = None
    raw_tensor: Any = None
    write_requested: bool = False
    commit_requested: bool = False
    production_activation_requested: bool = False
    def validate(self):
        if not self.request_id: raise ValueError("request_id required")
        if self.real_payload is not None: raise ValueError("real payload forbidden")
        if self.raw_tensor is not None: raise ValueError("raw tensor forbidden")
        self.envelope.validate()
        return self

@dataclass(frozen=True)
class QSpinSyntheticPayloadExecutionDecision:
    status: QSpinSyntheticPayloadStatus
    executed: bool
    block_reasons: Tuple[QSpinSyntheticPayloadBlockReason, ...] = ()
    def validate(self):
        if self.status is QSpinSyntheticPayloadStatus.EXECUTED_SYNTHETIC_ONLY and (not self.executed or self.block_reasons): raise ValueError("bad synthetic execution decision")
        if self.status is QSpinSyntheticPayloadStatus.BLOCKED and (self.executed or not self.block_reasons): raise ValueError("bad blocked synthetic execution decision")
        return self

@dataclass(frozen=True)
class QSpinSyntheticPayloadExecutionTrace:
    request_id: str
    status: QSpinSyntheticPayloadStatus
    safe_summary: Mapping[str, Any]
    def to_dict(self): return {"request_id":self.request_id,"status":self.status.value,"safe_summary":dict(self.safe_summary)}

@dataclass(frozen=True)
class QSpinSyntheticPayloadAuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    secret_free: bool = True
    raw_payload_free: bool = True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit event fields required")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe audit event")
        return self

@dataclass(frozen=True)
class QSpinSyntheticPayloadExecutionResult:
    request: QSpinSyntheticPayloadExecutionRequest
    decision: QSpinSyntheticPayloadExecutionDecision
    trace: QSpinSyntheticPayloadExecutionTrace
    audit_event: QSpinSyntheticPayloadAuditEvent
    transferred_payload: bool = False
    stored_payload: bool = False
    wrote_state: bool = False
    executed_commit: bool = False
    production_activated: bool = False
    def validate(self):
        self.decision.validate(); self.audit_event.validate()
        if any([self.transferred_payload,self.stored_payload,self.wrote_state,self.executed_commit,self.production_activated]):
            raise ValueError("synthetic payload harness must not produce live effects")
        return self

class QSpinSyntheticPayloadExecutionHarness:
    def __init__(self, policy: Optional[QSpinSyntheticPayloadGeneratorPolicy]=None):
        self.policy=(policy or build_default_qspin_synthetic_payload_generator_policy()).validate()
    def execute(self, request: QSpinSyntheticPayloadExecutionRequest) -> QSpinSyntheticPayloadExecutionResult:
        reasons=[]
        try: request.validate()
        except ValueError as exc:
            msg=str(exc)
            if "real payload" in msg: reasons.append(QSpinSyntheticPayloadBlockReason.REAL_PAYLOAD_PRESENT)
            elif "raw tensor" in msg: reasons.append(QSpinSyntheticPayloadBlockReason.RAW_TENSOR_PRESENT)
            elif "shape" in msg: reasons.append(QSpinSyntheticPayloadBlockReason.UNSAFE_SHAPE)
            elif "budget" in msg: reasons.append(QSpinSyntheticPayloadBlockReason.BUDGET_EXCEEDED)
            elif "norm" in msg: reasons.append(QSpinSyntheticPayloadBlockReason.NORM_BAND_INVALID)
            else: reasons.append(QSpinSyntheticPayloadBlockReason.MODE_DISABLED)
        checks=[(request.roundtrip_approved,QSpinSyntheticPayloadBlockReason.MISSING_ROUNDTRIP_APPROVAL),(request.permission_approved,QSpinSyntheticPayloadBlockReason.MISSING_PERMISSION_APPROVAL),(request.executor_approved,QSpinSyntheticPayloadBlockReason.MISSING_EXECUTOR_APPROVAL),(request.payload_dry_run_approved,QSpinSyntheticPayloadBlockReason.MISSING_PAYLOAD_DRY_RUN_APPROVAL),(request.dispatch_approved,QSpinSyntheticPayloadBlockReason.MISSING_DISPATCH_APPROVAL),(request.kill_switch_allows,QSpinSyntheticPayloadBlockReason.MISSING_KILL_SWITCH_APPROVAL),(request.rollback_evidence_present,QSpinSyntheticPayloadBlockReason.MISSING_ROLLBACK_EVIDENCE),(request.commit_gate_approved,QSpinSyntheticPayloadBlockReason.MISSING_COMMIT_GATE_APPROVAL)]
        for ok,reason in checks:
            if not ok: reasons.append(reason)
        if request.write_requested: reasons.append(QSpinSyntheticPayloadBlockReason.WRITE_REQUESTED)
        if request.commit_requested: reasons.append(QSpinSyntheticPayloadBlockReason.COMMIT_REQUESTED)
        if request.production_activation_requested: reasons.append(QSpinSyntheticPayloadBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if reasons:
            decision=QSpinSyntheticPayloadExecutionDecision(QSpinSyntheticPayloadStatus.BLOCKED,False,tuple(dict.fromkeys(reasons))).validate()
        else:
            decision=QSpinSyntheticPayloadExecutionDecision(QSpinSyntheticPayloadStatus.EXECUTED_SYNTHETIC_ONLY,True).validate()
        summary=request.envelope.to_safe_dict() if isinstance(request.envelope,QSpinSyntheticPayloadEnvelope) else {"invalid_envelope":True}
        summary["block_count"]=len(decision.block_reasons)
        trace=QSpinSyntheticPayloadExecutionTrace(request.request_id,decision.status,summary)
        audit=QSpinSyntheticPayloadAuditEvent("audit_"+request.request_id,"synthetic_payload_execute",tuple(r.value for r in decision.block_reasons)).validate()
        return QSpinSyntheticPayloadExecutionResult(request,decision,trace,audit).validate()

def build_default_qspin_synthetic_payload_generator_policy(): return QSpinSyntheticPayloadGeneratorPolicy().validate()
def build_default_qspin_synthetic_payload_execution_harness(): return QSpinSyntheticPayloadExecutionHarness()
