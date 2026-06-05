"""QSPIN-PROD-2 guarded bridge dispatch simulation.

Simulates bridge route selection after PROD-1 gates, payload dry-run, and shadow
bus dispatch have approved simulation metadata. It never performs live routing
or writes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

class QSpinGuardedDispatchMode(str, Enum):
    DISABLED="disabled"
    SIMULATION_ONLY="simulation_only"
class QSpinGuardedDispatchStatus(str, Enum):
    SIMULATED="simulated"
    BLOCKED="blocked"
class QSpinGuardedDispatchBlockReason(str, Enum):
    MODE_DISABLED="mode_disabled"
    UNKNOWN_BRIDGE_KIND="unknown_bridge_kind"
    PAYLOAD_DRY_RUN_MISSING="payload_dry_run_missing"
    SHADOW_BUS_MISSING="shadow_bus_missing"
    COMMIT_GATE_MISSING="commit_gate_missing"
    ROLLBACK_MISSING="rollback_missing"
    SOURCE_MATRIX_MISSING="source_matrix_missing"
    LIVE_ROUTING_REQUESTED="live_routing_requested"
    PAYLOAD_TRANSFER_REQUESTED="payload_transfer_requested"
    WRITE_REQUESTED="write_requested"
    COMMIT_REQUESTED="commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED="production_activation_requested"
class QSpinBridgeDispatchKind(str, Enum):
    WM8_INTERNAL="wm8_internal"
    MAP_6_TO_8="map_6_to_8"
    MAP_8_TO_10="map_8_to_10"
    VALIDATOR_6_TO_10="validator_6_to_10"
    DEPTH_BRIDGE="depth_bridge"
    PHASE_BRIDGE="phase_bridge"

@dataclass(frozen=True)
class QSpinGuardedDispatchPolicy:
    mode: QSpinGuardedDispatchMode=QSpinGuardedDispatchMode.SIMULATION_ONLY
    allow_live_routing: bool=False
    allow_payload_transfer: bool=False
    allow_writes: bool=False
    allow_commits: bool=False
    allow_production_activation: bool=False
    def validate(self):
        if self.mode is QSpinGuardedDispatchMode.DISABLED: raise ValueError("guarded dispatch disabled")
        if self.allow_live_routing or self.allow_payload_transfer or self.allow_writes or self.allow_commits or self.allow_production_activation:
            raise ValueError("unsafe guarded dispatch policy")
        return self
@dataclass(frozen=True)
class QSpinBridgeDispatchRouteSummary:
    bridge_kind: QSpinBridgeDispatchKind
    source_map: str
    target_map: str
    route_hops: Tuple[str,...]
    def validate(self):
        if not isinstance(self.bridge_kind,QSpinBridgeDispatchKind): raise ValueError("invalid bridge kind")
        if not self.source_map or not self.target_map or not self.route_hops: raise ValueError("route summary incomplete")
        return self
    def to_dict(self): return {"bridge_kind":self.bridge_kind.value,"source_map":self.source_map,"target_map":self.target_map,"route_hops":list(self.route_hops)}
@dataclass(frozen=True)
class QSpinBridgeDispatchSimulationRequest:
    request_id: str
    route: QSpinBridgeDispatchRouteSummary
    payload_dry_run_approved: bool
    shadow_bus_approved: bool
    commit_gate_approved: bool
    rollback_evidence_present: bool
    source_matrix_complete: bool
    live_routing_requested: bool=False
    payload_transfer_requested: bool=False
    write_requested: bool=False
    commit_requested: bool=False
    production_activation_requested: bool=False
    def validate(self):
        if not self.request_id: raise ValueError("request_id required")
        self.route.validate(); return self
@dataclass(frozen=True)
class QSpinBridgeDispatchSimulationDecision:
    status: QSpinGuardedDispatchStatus
    simulated: bool
    block_reasons: Tuple[QSpinGuardedDispatchBlockReason,...]=()
    def validate(self):
        if self.status is QSpinGuardedDispatchStatus.SIMULATED and (not self.simulated or self.block_reasons): raise ValueError("bad simulated decision")
        if self.status is QSpinGuardedDispatchStatus.BLOCKED and (self.simulated or not self.block_reasons): raise ValueError("bad blocked decision")
        return self
@dataclass(frozen=True)
class QSpinBridgeDispatchSafetyReport:
    live_routing: bool=False
    payload_transfer: bool=False
    writes: bool=False
    commits: bool=False
    production_activation: bool=False
    def validate(self):
        if any([self.live_routing,self.payload_transfer,self.writes,self.commits,self.production_activation]): raise ValueError("unsafe dispatch safety report")
        return self
@dataclass(frozen=True)
class QSpinBridgeDispatchSimulationResult:
    request: QSpinBridgeDispatchSimulationRequest
    decision: QSpinBridgeDispatchSimulationDecision
    route_summary: Mapping[str, Any]
    safety_report: QSpinBridgeDispatchSafetyReport=field(default_factory=QSpinBridgeDispatchSafetyReport)
    def validate(self):
        self.decision.validate(); self.safety_report.validate(); return self
    def to_dict(self): return {"request_id":self.request.request_id,"decision":{"status":self.decision.status.value,"simulated":self.decision.simulated,"block_reasons":[r.value for r in self.decision.block_reasons]},"route_summary":dict(self.route_summary)}
class QSpinGuardedBridgeDispatchSimulator:
    def __init__(self, policy: Optional[QSpinGuardedDispatchPolicy]=None): self.policy=(policy or build_default_qspin_guarded_dispatch_policy()).validate()
    def simulate(self, request: QSpinBridgeDispatchSimulationRequest) -> QSpinBridgeDispatchSimulationResult:
        request.validate(); reasons=[]
        if not request.payload_dry_run_approved: reasons.append(QSpinGuardedDispatchBlockReason.PAYLOAD_DRY_RUN_MISSING)
        if not request.shadow_bus_approved: reasons.append(QSpinGuardedDispatchBlockReason.SHADOW_BUS_MISSING)
        if not request.commit_gate_approved: reasons.append(QSpinGuardedDispatchBlockReason.COMMIT_GATE_MISSING)
        if not request.rollback_evidence_present: reasons.append(QSpinGuardedDispatchBlockReason.ROLLBACK_MISSING)
        if not request.source_matrix_complete: reasons.append(QSpinGuardedDispatchBlockReason.SOURCE_MATRIX_MISSING)
        if request.live_routing_requested: reasons.append(QSpinGuardedDispatchBlockReason.LIVE_ROUTING_REQUESTED)
        if request.payload_transfer_requested: reasons.append(QSpinGuardedDispatchBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.write_requested: reasons.append(QSpinGuardedDispatchBlockReason.WRITE_REQUESTED)
        if request.commit_requested: reasons.append(QSpinGuardedDispatchBlockReason.COMMIT_REQUESTED)
        if request.production_activation_requested: reasons.append(QSpinGuardedDispatchBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if reasons: dec=QSpinBridgeDispatchSimulationDecision(QSpinGuardedDispatchStatus.BLOCKED, False, tuple(dict.fromkeys(reasons))).validate()
        else: dec=QSpinBridgeDispatchSimulationDecision(QSpinGuardedDispatchStatus.SIMULATED, True).validate()
        return QSpinBridgeDispatchSimulationResult(request,dec,request.route.to_dict()).validate()
def build_default_qspin_guarded_dispatch_policy(): return QSpinGuardedDispatchPolicy().validate()
