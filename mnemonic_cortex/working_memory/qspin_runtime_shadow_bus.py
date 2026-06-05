"""QSPIN-PROD-2 runtime adapter shadow bus.

The shadow bus registers adapter metadata and simulates dispatch only. It never
calls live QD6A runtime modules, mutates state, transfers payloads, writes
shared slots/external memory/QH storage, executes commits, or activates
production.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib, json

class QSpinShadowBusMode(str, Enum):
    DISABLED="disabled"
    SIMULATION_ONLY="simulation_only"
class QSpinShadowBusStatus(str, Enum):
    SIMULATED="simulated"
    BLOCKED="blocked"
    IDEMPOTENT_REPLAY="idempotent_replay"
class QSpinShadowBusBlockReason(str, Enum):
    BUS_DISABLED="bus_disabled"
    ADAPTER_NOT_REGISTERED="adapter_not_registered"
    SOURCE_MATRIX_MISSING="source_matrix_missing"
    ROLLBACK_MISSING="rollback_missing"
    KILL_SWITCH_BLOCK="kill_switch_block"
    COMMIT_GATE_BLOCK="commit_gate_block"
    SHADOW_ACTIVATION_MISSING="shadow_activation_missing"
    PAYLOAD_SUMMARY_MISSING="payload_summary_missing"
    LIVE_ROUTING_REQUESTED="live_routing_requested"
    PAYLOAD_TRANSFER_REQUESTED="payload_transfer_requested"
    WRITE_REQUESTED="write_requested"
    COMMIT_REQUESTED="commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED="production_activation_requested"

class QSpinShadowAdapterKind(str, Enum):
    WM8_INTERNAL="wm8_internal"
    MAP_6_TO_8="map_6_to_8"
    MAP_8_TO_10="map_8_to_10"
    VALIDATOR_6_TO_10="validator_6_to_10"
    DEPTH_BRIDGE="depth_bridge"
    PHASE_BRIDGE="phase_bridge"

@dataclass(frozen=True)
class QSpinShadowBusConfig:
    mode: QSpinShadowBusMode = QSpinShadowBusMode.SIMULATION_ONLY
    allow_live_routing: bool=False
    allow_payload_transfer: bool=False
    allow_writes: bool=False
    allow_commit_execution: bool=False
    allow_production_activation: bool=False
    require_source_matrix: bool=True
    require_rollback: bool=True
    require_kill_switch: bool=True
    require_commit_gate: bool=True
    require_shadow_activation: bool=True
    require_payload_summary: bool=True
    def validate(self):
        if self.mode is QSpinShadowBusMode.DISABLED: raise ValueError("shadow bus disabled")
        if self.allow_live_routing: raise ValueError("live routing forbidden")
        if self.allow_payload_transfer: raise ValueError("payload transfer forbidden")
        if self.allow_writes: raise ValueError("writes forbidden")
        if self.allow_commit_execution: raise ValueError("commit execution forbidden")
        if self.allow_production_activation: raise ValueError("production activation forbidden")
        return self

@dataclass(frozen=True)
class QSpinShadowAdapterEndpoint:
    endpoint_id: str
    adapter_kind: QSpinShadowAdapterKind
    source_scope: str
    target_scope: str
    metadata: Mapping[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.endpoint_id or not self.source_scope or not self.target_scope: raise ValueError("adapter endpoint fields required")
        if not isinstance(self.adapter_kind,QSpinShadowAdapterKind): raise ValueError("invalid adapter kind")
        return self

@dataclass(frozen=True)
class QSpinShadowAdapterRegistration:
    adapter_id: str
    endpoint: QSpinShadowAdapterEndpoint
    metadata_only: bool=True
    mutates_runtime: bool=False
    calls_live_module: bool=False
    def validate(self):
        if not self.adapter_id: raise ValueError("adapter_id required")
        self.endpoint.validate()
        if not self.metadata_only or self.mutates_runtime or self.calls_live_module:
            raise ValueError("shadow adapter must be metadata-only and non-mutating")
        return self

@dataclass(frozen=True)
class QSpinShadowBusMessage:
    message_id: str
    source_adapter_id: str
    target_adapter_id: str
    payload_summary_hash: str
    metadata: Mapping[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.message_id or not self.source_adapter_id or not self.target_adapter_id or not self.payload_summary_hash:
            raise ValueError("message fields required")
        return self

@dataclass(frozen=True)
class QSpinShadowBusDispatchRequest:
    request_id: str
    message: QSpinShadowBusMessage
    source_matrix_complete: bool
    rollback_evidence_present: bool
    kill_switch_allows: bool
    commit_gate_allows: bool
    shadow_activation_allows: bool
    payload_summary_approved: bool
    live_routing_requested: bool=False
    payload_transfer_requested: bool=False
    write_requested: bool=False
    commit_requested: bool=False
    production_activation_requested: bool=False
    def validate(self):
        if not self.request_id: raise ValueError("request_id required")
        self.message.validate(); return self
    def key(self):
        data={"request_id":self.request_id,"message":self.message.message_id,"source":self.message.source_adapter_id,"target":self.message.target_adapter_id,"hash":self.message.payload_summary_hash}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class QSpinShadowBusDispatchDecision:
    status: QSpinShadowBusStatus
    allowed_simulation: bool
    block_reasons: Tuple[QSpinShadowBusBlockReason,...]=()
    def validate(self):
        if self.status is QSpinShadowBusStatus.SIMULATED and (not self.allowed_simulation or self.block_reasons): raise ValueError("bad simulated decision")
        if self.status is QSpinShadowBusStatus.BLOCKED and (self.allowed_simulation or not self.block_reasons): raise ValueError("bad blocked decision")
        return self

@dataclass(frozen=True)
class QSpinShadowBusTrace:
    request_id: str
    status: QSpinShadowBusStatus
    safe_summary: Mapping[str, Any]
    def to_dict(self): return {"request_id":self.request_id,"status":self.status.value,"safe_summary":dict(self.safe_summary)}
@dataclass(frozen=True)
class QSpinShadowBusAuditEvent:
    event_id: str
    action: str
    safe_summary: Mapping[str, Any]
    secret_free: bool=True
    raw_payload_free: bool=True
    def validate(self):
        if not self.event_id or not self.action: raise ValueError("audit fields required")
        if not self.secret_free or not self.raw_payload_free: raise ValueError("unsafe audit event")
        return self
@dataclass(frozen=True)
class QSpinShadowBusDispatchResult:
    request: QSpinShadowBusDispatchRequest
    decision: QSpinShadowBusDispatchDecision
    trace: QSpinShadowBusTrace
    audit_event: QSpinShadowBusAuditEvent
    routed_live_data: bool=False
    transferred_payload: bool=False
    wrote_state: bool=False
    executed_commit: bool=False
    production_activated: bool=False
    def validate(self):
        self.decision.validate(); self.audit_event.validate()
        if any([self.routed_live_data,self.transferred_payload,self.wrote_state,self.executed_commit,self.production_activated]):
            raise ValueError("shadow bus dispatch must not create live effects")
        return self

class QSpinRuntimeAdapterShadowBus:
    def __init__(self, config: Optional[QSpinShadowBusConfig]=None):
        self.config=(config or build_default_qspin_shadow_bus_config()).validate()
        self.adapters: Dict[str,QSpinShadowAdapterRegistration]={}
        self.history: Dict[str,QSpinShadowBusDispatchResult]={}
    def register_adapter(self, registration: QSpinShadowAdapterRegistration):
        registration.validate()
        if registration.adapter_id in self.adapters: raise ValueError("duplicate adapter registration")
        self.adapters[registration.adapter_id]=registration
    def dispatch(self, request: QSpinShadowBusDispatchRequest) -> QSpinShadowBusDispatchResult:
        request.validate(); key=request.key()
        if key in self.history:
            old=self.history[key]
            dec=QSpinShadowBusDispatchDecision(QSpinShadowBusStatus.IDEMPOTENT_REPLAY, old.decision.allowed_simulation, old.decision.block_reasons).validate()
            tr=QSpinShadowBusTrace(request.request_id, dec.status, {"idempotent_replay": True, "prior_status": old.decision.status.value})
            ev=QSpinShadowBusAuditEvent("audit_"+request.request_id, "idempotent_replay", tr.safe_summary).validate()
            return QSpinShadowBusDispatchResult(request, dec, tr, ev).validate()
        reasons=[]
        if request.message.source_adapter_id not in self.adapters or request.message.target_adapter_id not in self.adapters: reasons.append(QSpinShadowBusBlockReason.ADAPTER_NOT_REGISTERED)
        if not request.source_matrix_complete: reasons.append(QSpinShadowBusBlockReason.SOURCE_MATRIX_MISSING)
        if not request.rollback_evidence_present: reasons.append(QSpinShadowBusBlockReason.ROLLBACK_MISSING)
        if not request.kill_switch_allows: reasons.append(QSpinShadowBusBlockReason.KILL_SWITCH_BLOCK)
        if not request.commit_gate_allows: reasons.append(QSpinShadowBusBlockReason.COMMIT_GATE_BLOCK)
        if not request.shadow_activation_allows: reasons.append(QSpinShadowBusBlockReason.SHADOW_ACTIVATION_MISSING)
        if not request.payload_summary_approved: reasons.append(QSpinShadowBusBlockReason.PAYLOAD_SUMMARY_MISSING)
        if request.live_routing_requested: reasons.append(QSpinShadowBusBlockReason.LIVE_ROUTING_REQUESTED)
        if request.payload_transfer_requested: reasons.append(QSpinShadowBusBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.write_requested: reasons.append(QSpinShadowBusBlockReason.WRITE_REQUESTED)
        if request.commit_requested: reasons.append(QSpinShadowBusBlockReason.COMMIT_REQUESTED)
        if request.production_activation_requested: reasons.append(QSpinShadowBusBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if reasons: dec=QSpinShadowBusDispatchDecision(QSpinShadowBusStatus.BLOCKED, False, tuple(dict.fromkeys(reasons))).validate()
        else: dec=QSpinShadowBusDispatchDecision(QSpinShadowBusStatus.SIMULATED, True).validate()
        tr=QSpinShadowBusTrace(request.request_id, dec.status, {"source_adapter":request.message.source_adapter_id,"target_adapter":request.message.target_adapter_id,"payload_summary_hash":request.message.payload_summary_hash,"block_count":len(dec.block_reasons)})
        ev=QSpinShadowBusAuditEvent("audit_"+request.request_id,"simulate_dispatch",tr.safe_summary).validate()
        res=QSpinShadowBusDispatchResult(request,dec,tr,ev).validate(); self.history[key]=res; return res

def build_default_qspin_shadow_bus_config(): return QSpinShadowBusConfig().validate()
def validate_qspin_shadow_bus_config(config: QSpinShadowBusConfig): return config.validate()
