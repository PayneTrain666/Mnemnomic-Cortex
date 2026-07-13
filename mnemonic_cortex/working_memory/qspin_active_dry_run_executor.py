"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-3 feature-flagged active-dry-run bridge executor.

The executor shape exists here, but it is active-dry-run simulation only. It
never calls live QD6A runtime modules, routes live data, transfers real payloads,
writes state, executes commits, or activates production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib
import json


class QSpinActiveDryRunMode(str, Enum):
    DISABLED = "disabled"
    ACTIVE_DRY_RUN_ONLY = "active_dry_run_only"


class QSpinActiveDryRunStatus(str, Enum):
    EXECUTED_ACTIVE_DRY_RUN = "executed_active_dry_run"
    BLOCKED = "blocked"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class QSpinActiveDryRunBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    FEATURE_GATE_MISSING = "feature_gate_missing"
    SOURCE_MATRIX_MISSING = "source_matrix_missing"
    ROLLBACK_MISSING = "rollback_missing"
    KILL_SWITCH_BLOCK = "kill_switch_block"
    COMMIT_APPROVAL_MISSING = "commit_approval_missing"
    PAYLOAD_ROUNDTRIP_MISSING = "payload_roundtrip_missing"
    PERMISSION_DRY_RUN_MISSING = "permission_dry_run_missing"
    GUARDED_DISPATCH_MISSING = "guarded_dispatch_missing"
    PAYLOAD_SUMMARY_MISSING = "payload_summary_missing"
    LIVE_ROUTING_REQUESTED = "live_routing_requested"
    PAYLOAD_TRANSFER_REQUESTED = "payload_transfer_requested"
    WRITE_REQUESTED = "write_requested"
    COMMIT_REQUESTED = "commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"


@dataclass(frozen=True)
class QSpinActiveDryRunFeatureGate:
    active_dry_run_enabled: bool = False
    production_activation_enabled: bool = False

    def validate(self) -> "QSpinActiveDryRunFeatureGate":
        if self.production_activation_enabled:
            raise ValueError("production activation feature gate forbidden in PROD-3")
        return self


@dataclass(frozen=True)
class QSpinActiveDryRunExecutorConfig:
    mode: QSpinActiveDryRunMode = QSpinActiveDryRunMode.ACTIVE_DRY_RUN_ONLY
    allow_live_routing: bool = False
    allow_payload_transfer: bool = False
    allow_writes: bool = False
    allow_commits: bool = False
    allow_production_activation: bool = False
    require_feature_gate: bool = True
    require_source_matrix: bool = True
    require_rollback: bool = True
    require_kill_switch: bool = True
    require_commit_approval: bool = True
    require_payload_roundtrip: bool = True
    require_permission_dry_run: bool = True
    require_guarded_dispatch: bool = True
    require_payload_summary: bool = True

    def validate(self) -> "QSpinActiveDryRunExecutorConfig":
        if self.mode is QSpinActiveDryRunMode.DISABLED:
            raise ValueError("active-dry-run executor disabled")
        if any([self.allow_live_routing, self.allow_payload_transfer, self.allow_writes, self.allow_commits, self.allow_production_activation]):
            raise ValueError("unsafe active-dry-run executor config")
        for name in (
            "require_feature_gate", "require_source_matrix", "require_rollback",
            "require_kill_switch", "require_commit_approval", "require_payload_roundtrip",
            "require_permission_dry_run", "require_guarded_dispatch", "require_payload_summary",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be true")
        return self


@dataclass(frozen=True)
class QSpinActiveDryRunExecutionRequest:
    request_id: str
    bridge_plan_id: str
    feature_gate: QSpinActiveDryRunFeatureGate
    source_matrix_complete: bool
    rollback_evidence_present: bool
    kill_switch_allows: bool
    commit_approval_allows: bool
    payload_roundtrip_approved: bool
    permission_dry_run_approved: bool
    guarded_dispatch_approved: bool
    trace_safe_payload_summary_present: bool
    live_routing_requested: bool = False
    payload_transfer_requested: bool = False
    write_requested: bool = False
    commit_requested: bool = False
    production_activation_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinActiveDryRunExecutionRequest":
        if not self.request_id or not self.bridge_plan_id:
            raise ValueError("request_id and bridge_plan_id are required")
        self.feature_gate.validate()
        return self

    def key(self) -> str:
        data = {
            "request_id": self.request_id,
            "bridge_plan_id": self.bridge_plan_id,
            "source_matrix_complete": self.source_matrix_complete,
            "rollback_evidence_present": self.rollback_evidence_present,
            "commit_approval_allows": self.commit_approval_allows,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class QSpinActiveDryRunExecutionDecision:
    status: QSpinActiveDryRunStatus
    executed_active_dry_run: bool
    block_reasons: Tuple[QSpinActiveDryRunBlockReason, ...] = ()

    def validate(self) -> "QSpinActiveDryRunExecutionDecision":
        if self.status is QSpinActiveDryRunStatus.EXECUTED_ACTIVE_DRY_RUN and (not self.executed_active_dry_run or self.block_reasons):
            raise ValueError("executed active-dry-run decision malformed")
        if self.status is QSpinActiveDryRunStatus.BLOCKED and (self.executed_active_dry_run or not self.block_reasons):
            raise ValueError("blocked active-dry-run decision malformed")
        return self


@dataclass(frozen=True)
class QSpinActiveDryRunExecutionTrace:
    request_id: str
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "safe_summary": dict(self.safe_summary)}


@dataclass(frozen=True)
class QSpinActiveDryRunAuditEvent:
    event_id: str
    action: str
    reason_codes: Tuple[str, ...] = ()
    secret_free: bool = True
    raw_payload_free: bool = True

    def validate(self) -> "QSpinActiveDryRunAuditEvent":
        if not self.event_id or not self.action:
            raise ValueError("audit event requires ID and action")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("audit event must be trace-safe")
        return self


@dataclass(frozen=True)
class QSpinActiveDryRunExecutionResult:
    request: QSpinActiveDryRunExecutionRequest
    decision: QSpinActiveDryRunExecutionDecision
    trace: QSpinActiveDryRunExecutionTrace
    audit_event: QSpinActiveDryRunAuditEvent
    called_live_runtime: bool = False
    routed_live_data: bool = False
    transferred_payload: bool = False
    wrote_state: bool = False
    executed_commit: bool = False
    production_activated: bool = False

    def validate(self) -> "QSpinActiveDryRunExecutionResult":
        self.decision.validate()
        self.audit_event.validate()
        if any([
            self.called_live_runtime,
            self.routed_live_data,
            self.transferred_payload,
            self.wrote_state,
            self.executed_commit,
            self.production_activated,
        ]):
            raise ValueError("active-dry-run executor must not perform live effects")
        return self


class QSpinActiveDryRunBridgeExecutor:
    def __init__(self, config: Optional[QSpinActiveDryRunExecutorConfig] = None):
        self.config = (config or build_default_qspin_active_dry_run_executor_config()).validate()
        self.history: Dict[str, QSpinActiveDryRunExecutionResult] = {}

    def execute(self, request: QSpinActiveDryRunExecutionRequest) -> QSpinActiveDryRunExecutionResult:
        request.validate()
        key = request.key()
        if key in self.history:
            old = self.history[key]
            decision = QSpinActiveDryRunExecutionDecision(
                QSpinActiveDryRunStatus.IDEMPOTENT_REPLAY,
                old.decision.executed_active_dry_run,
                old.decision.block_reasons,
            ).validate()
            trace = QSpinActiveDryRunExecutionTrace(request.request_id, {"idempotent_replay": True, "prior_status": old.decision.status.value})
            event = QSpinActiveDryRunAuditEvent("audit_" + request.request_id, "idempotent_replay").validate()
            return QSpinActiveDryRunExecutionResult(request, decision, trace, event).validate()

        reasons = []
        if not request.feature_gate.active_dry_run_enabled:
            reasons.append(QSpinActiveDryRunBlockReason.FEATURE_GATE_MISSING)
        if not request.source_matrix_complete:
            reasons.append(QSpinActiveDryRunBlockReason.SOURCE_MATRIX_MISSING)
        if not request.rollback_evidence_present:
            reasons.append(QSpinActiveDryRunBlockReason.ROLLBACK_MISSING)
        if not request.kill_switch_allows:
            reasons.append(QSpinActiveDryRunBlockReason.KILL_SWITCH_BLOCK)
        if not request.commit_approval_allows:
            reasons.append(QSpinActiveDryRunBlockReason.COMMIT_APPROVAL_MISSING)
        if not request.payload_roundtrip_approved:
            reasons.append(QSpinActiveDryRunBlockReason.PAYLOAD_ROUNDTRIP_MISSING)
        if not request.permission_dry_run_approved:
            reasons.append(QSpinActiveDryRunBlockReason.PERMISSION_DRY_RUN_MISSING)
        if not request.guarded_dispatch_approved:
            reasons.append(QSpinActiveDryRunBlockReason.GUARDED_DISPATCH_MISSING)
        if not request.trace_safe_payload_summary_present:
            reasons.append(QSpinActiveDryRunBlockReason.PAYLOAD_SUMMARY_MISSING)
        if request.live_routing_requested:
            reasons.append(QSpinActiveDryRunBlockReason.LIVE_ROUTING_REQUESTED)
        if request.payload_transfer_requested:
            reasons.append(QSpinActiveDryRunBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.write_requested:
            reasons.append(QSpinActiveDryRunBlockReason.WRITE_REQUESTED)
        if request.commit_requested:
            reasons.append(QSpinActiveDryRunBlockReason.COMMIT_REQUESTED)
        if request.production_activation_requested:
            reasons.append(QSpinActiveDryRunBlockReason.PRODUCTION_ACTIVATION_REQUESTED)

        if reasons:
            decision = QSpinActiveDryRunExecutionDecision(
                QSpinActiveDryRunStatus.BLOCKED,
                False,
                tuple(dict.fromkeys(reasons)),
            ).validate()
        else:
            decision = QSpinActiveDryRunExecutionDecision(
                QSpinActiveDryRunStatus.EXECUTED_ACTIVE_DRY_RUN,
                True,
            ).validate()

        trace = QSpinActiveDryRunExecutionTrace(
            request.request_id,
            {
                "bridge_plan_id": request.bridge_plan_id,
                "block_count": len(decision.block_reasons),
                "active_dry_run_only": True,
                "live_effects": False,
            },
        )
        event = QSpinActiveDryRunAuditEvent(
            "audit_" + request.request_id,
            "active_dry_run_execute",
            tuple(r.value for r in decision.block_reasons),
        ).validate()
        result = QSpinActiveDryRunExecutionResult(request, decision, trace, event).validate()
        self.history[key] = result
        return result


def build_default_qspin_active_dry_run_executor_config() -> QSpinActiveDryRunExecutorConfig:
    return QSpinActiveDryRunExecutorConfig().validate()


def validate_qspin_active_dry_run_executor_config(config: QSpinActiveDryRunExecutorConfig) -> QSpinActiveDryRunExecutorConfig:
    return config.validate()
