"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-2 trace-safe payload dry-run path.

This module accepts payload *metadata* only. It rejects raw payload/tensor fields,
performs deterministic metadata summarisation, checks declared shape/budget, and
never transfers, stores, logs, or writes payload data.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib, json

class QSpinPayloadDryRunMode(str, Enum):
    DISABLED = "disabled"
    SUMMARY_ONLY = "summary_only"

class QSpinPayloadDryRunStatus(str, Enum):
    APPROVED_SUMMARY_ONLY = "approved_summary_only"
    BLOCKED = "blocked"

class QSpinPayloadDryRunBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    RAW_PAYLOAD_PRESENT = "raw_payload_present"
    RAW_TENSOR_PRESENT = "raw_tensor_present"
    UNSAFE_SHAPE = "unsafe_shape"
    BUDGET_EXCEEDED = "budget_exceeded"
    MISSING_SOURCE_OR_TARGET = "missing_source_or_target"
    UNSAFE_NORM_BAND = "unsafe_norm_band"

@dataclass(frozen=True)
class QSpinPayloadDryRunShapeSummary:
    dims: Tuple[int, ...]
    dtype: str = "float32"
    max_rank: int = 4
    max_elements: int = 1_000_000
    def element_count(self) -> int:
        n=1
        for d in self.dims: n*=d
        return n
    def validate(self) -> "QSpinPayloadDryRunShapeSummary":
        if not self.dims or len(self.dims)>self.max_rank:
            raise ValueError("unsafe payload shape rank")
        if any((not isinstance(d,int)) or d<=0 for d in self.dims):
            raise ValueError("payload shape dimensions must be positive integers")
        if self.element_count()>self.max_elements:
            raise ValueError("payload shape exceeds max_elements")
        if not self.dtype or any(ch in self.dtype for ch in "\n\r\t"):
            raise ValueError("unsafe dtype string")
        return self
    def to_dict(self):
        return {"dims": list(self.dims), "dtype": self.dtype, "element_count": self.element_count()}

@dataclass(frozen=True)
class QSpinPayloadDryRunBudgetSummary:
    declared_bytes: int
    max_bytes: int = 16_777_216
    bandwidth_units: int = 1
    max_bandwidth_units: int = 16
    def validate(self) -> "QSpinPayloadDryRunBudgetSummary":
        if self.declared_bytes < 0 or self.max_bytes <= 0:
            raise ValueError("invalid payload byte budget")
        if self.declared_bytes > self.max_bytes:
            raise ValueError("payload budget exceeded")
        if self.bandwidth_units < 0 or self.bandwidth_units > self.max_bandwidth_units:
            raise ValueError("payload bandwidth budget exceeded")
        return self
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class QSpinPayloadDryRunEnvelopeSummary:
    payload_kind: str
    source_id: str
    target_id: str
    shape: QSpinPayloadDryRunShapeSummary
    budget: QSpinPayloadDryRunBudgetSummary
    norm_band: Tuple[float, float] = (0.0, 1.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def validate(self) -> "QSpinPayloadDryRunEnvelopeSummary":
        if not self.payload_kind or not self.source_id or not self.target_id:
            raise ValueError("payload_kind, source_id, and target_id are required")
        self.shape.validate(); self.budget.validate()
        lo,hi=self.norm_band
        if lo < 0 or hi < lo or hi > 1e6:
            raise ValueError("unsafe norm band")
        return self
    def safe_hash(self) -> str:
        data={"payload_kind":self.payload_kind,"source_id":self.source_id,"target_id":self.target_id,"shape":self.shape.to_dict(),"budget":self.budget.to_dict(),"norm_band":list(self.norm_band)}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    def to_dict(self):
        return {"payload_kind":self.payload_kind,"source_id":self.source_id,"target_id":self.target_id,"shape":self.shape.to_dict(),"budget":self.budget.to_dict(),"norm_band":list(self.norm_band),"safe_hash":self.safe_hash(),"metadata_keys":sorted(str(k) for k in self.metadata.keys())}

@dataclass(frozen=True)
class QSpinPayloadDryRunRequest:
    request_id: str
    envelope: Optional[QSpinPayloadDryRunEnvelopeSummary]
    mode: QSpinPayloadDryRunMode = QSpinPayloadDryRunMode.SUMMARY_ONLY
    raw_payload: Any = None
    raw_tensor: Any = None
    def validate(self) -> "QSpinPayloadDryRunRequest":
        if not self.request_id: raise ValueError("request_id is required")
        if not isinstance(self.mode,QSpinPayloadDryRunMode): raise ValueError("mode must be QSpinPayloadDryRunMode")
        if self.raw_payload is not None: raise ValueError("raw payload rejected")
        if self.raw_tensor is not None: raise ValueError("raw tensor rejected")
        if self.envelope is not None: self.envelope.validate()
        return self

@dataclass(frozen=True)
class QSpinPayloadDryRunDecision:
    status: QSpinPayloadDryRunStatus
    approved: bool
    block_reasons: Tuple[QSpinPayloadDryRunBlockReason, ...] = ()
    def validate(self):
        if self.status is QSpinPayloadDryRunStatus.APPROVED_SUMMARY_ONLY and (not self.approved or self.block_reasons):
            raise ValueError("approved payload dry-run cannot have block reasons")
        if self.status is QSpinPayloadDryRunStatus.BLOCKED and (self.approved or not self.block_reasons):
            raise ValueError("blocked payload dry-run requires reasons")
        return self

@dataclass(frozen=True)
class QSpinPayloadDryRunTrace:
    request_id: str
    status: QSpinPayloadDryRunStatus
    safe_summary: Mapping[str, Any]
    def to_dict(self): return {"request_id":self.request_id,"status":self.status.value,"safe_summary":dict(self.safe_summary)}

@dataclass(frozen=True)
class QSpinPayloadDryRunResult:
    request: QSpinPayloadDryRunRequest
    decision: QSpinPayloadDryRunDecision
    trace: QSpinPayloadDryRunTrace
    transferred_payload: bool=False
    stored_raw_payload: bool=False
    wrote_external_state: bool=False
    def validate(self):
        self.decision.validate()
        if self.transferred_payload or self.stored_raw_payload or self.wrote_external_state:
            raise ValueError("payload dry-run must not transfer/store/write payloads")
        return self
    def to_dict(self): return {"request_id":self.request.request_id,"decision":{"status":self.decision.status.value,"approved":self.decision.approved,"block_reasons":[r.value for r in self.decision.block_reasons]},"trace":self.trace.to_dict(),"transferred_payload":self.transferred_payload,"stored_raw_payload":self.stored_raw_payload,"wrote_external_state":self.wrote_external_state}

@dataclass(frozen=True)
class QSpinPayloadDryRunPolicy:
    mode: QSpinPayloadDryRunMode = QSpinPayloadDryRunMode.SUMMARY_ONLY
    allow_raw_payload: bool = False
    allow_raw_tensor: bool = False
    allow_transfer: bool = False
    allow_storage: bool = False
    allow_writes: bool = False
    def validate(self):
        if self.mode is QSpinPayloadDryRunMode.DISABLED: raise ValueError("payload dry-run policy disabled")
        if self.allow_raw_payload or self.allow_raw_tensor or self.allow_transfer or self.allow_storage or self.allow_writes:
            raise ValueError("unsafe payload dry-run policy")
        return self

class QSpinTraceSafePayloadSummarizer:
    def __init__(self, policy: Optional[QSpinPayloadDryRunPolicy]=None):
        self.policy=(policy or build_default_qspin_payload_dry_run_policy()).validate()
    def dry_run(self, request: QSpinPayloadDryRunRequest) -> QSpinPayloadDryRunResult:
        reasons=[]
        try: request.validate()
        except ValueError as exc:
            msg=str(exc)
            if "raw payload" in msg: reasons.append(QSpinPayloadDryRunBlockReason.RAW_PAYLOAD_PRESENT)
            elif "raw tensor" in msg: reasons.append(QSpinPayloadDryRunBlockReason.RAW_TENSOR_PRESENT)
            elif "shape" in msg: reasons.append(QSpinPayloadDryRunBlockReason.UNSAFE_SHAPE)
            elif "budget" in msg: reasons.append(QSpinPayloadDryRunBlockReason.BUDGET_EXCEEDED)
            elif "norm" in msg: reasons.append(QSpinPayloadDryRunBlockReason.UNSAFE_NORM_BAND)
            else: reasons.append(QSpinPayloadDryRunBlockReason.MISSING_SOURCE_OR_TARGET)
        if request.mode is QSpinPayloadDryRunMode.DISABLED: reasons.append(QSpinPayloadDryRunBlockReason.MODE_DISABLED)
        if request.envelope is None: reasons.append(QSpinPayloadDryRunBlockReason.MISSING_SOURCE_OR_TARGET)
        if reasons:
            decision=QSpinPayloadDryRunDecision(QSpinPayloadDryRunStatus.BLOCKED, False, tuple(dict.fromkeys(reasons))).validate()
            trace=QSpinPayloadDryRunTrace(request.request_id, decision.status, {"block_count": len(decision.block_reasons)})
        else:
            summary=request.envelope.to_dict()
            decision=QSpinPayloadDryRunDecision(QSpinPayloadDryRunStatus.APPROVED_SUMMARY_ONLY, True).validate()
            trace=QSpinPayloadDryRunTrace(request.request_id, decision.status, summary)
        return QSpinPayloadDryRunResult(request, decision, trace).validate()

def build_default_qspin_payload_dry_run_policy() -> QSpinPayloadDryRunPolicy:
    return QSpinPayloadDryRunPolicy().validate()
