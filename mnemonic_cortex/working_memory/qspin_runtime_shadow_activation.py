"""QSPIN-PROD-1 feature-flagged runtime shadow activation.

This module creates a shadow-only activation controller. It never routes live
QSPIN data, transfers payloads, writes state, writes external memory, writes QH
storage, or executes commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple
import hashlib

try:
    from .qspin_production_config import QSpinRuntimeFeatureFlag, QSpinProductionConfig, build_default_qspin_production_config
    from .qspin_commit_gate_dry_run import QSpinCommitGateDryRunResult
    from .qspin_kill_switch import QSpinKillSwitchDecision, QSpinKillSwitchState
    from .qspin_rollback_harness import QSpinRollbackDryRunResult
except Exception:  # pragma: no cover
    from qspin_production_config import QSpinRuntimeFeatureFlag, QSpinProductionConfig, build_default_qspin_production_config
    from qspin_commit_gate_dry_run import QSpinCommitGateDryRunResult
    from qspin_kill_switch import QSpinKillSwitchDecision, QSpinKillSwitchState
    from qspin_rollback_harness import QSpinRollbackDryRunResult


class QSpinShadowActivationMode(str, Enum):
    DISABLED = "disabled"
    SHADOW_ONLY = "shadow_only"


class QSpinShadowActivationStatus(str, Enum):
    ALLOWED_SHADOW_ONLY = "allowed_shadow_only"
    BLOCKED = "blocked"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class QSpinShadowActivationBlockReason(str, Enum):
    CONFIG_INVALID = "config_invalid"
    MODE_DISABLED = "mode_disabled"
    ACTIVE_ROUTING_REQUESTED = "active_routing_requested"
    PAYLOAD_TRANSFER_REQUESTED = "payload_transfer_requested"
    WRITE_REQUESTED = "write_requested"
    COMMIT_REQUESTED = "commit_requested"
    SOURCE_MATRIX_INCOMPLETE = "source_matrix_incomplete"
    ROLLBACK_EVIDENCE_MISSING = "rollback_evidence_missing"
    KILL_SWITCH_NOT_ALLOWING = "kill_switch_not_allowing"
    COMMIT_GATE_NOT_ALLOWING = "commit_gate_not_allowing"
    RAW_TRACE_REQUESTED = "raw_trace_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationConfig:
    config_id: str = "qspin_prod1_shadow_activation_config"
    mode: QSpinShadowActivationMode = QSpinShadowActivationMode.DISABLED
    allow_shadow_mode: bool = True
    allow_active_routing: bool = False
    allow_payload_transfer: bool = False
    allow_writes: bool = False
    allow_commit_execution: bool = False
    allow_raw_payload_trace: bool = False
    allow_production_activation: bool = False
    require_source_matrix: bool = True
    require_rollback_evidence: bool = True
    require_kill_switch: bool = True
    require_commit_gate_dry_run: bool = True

    def validate(self) -> "QSpinRuntimeShadowActivationConfig":
        if not self.config_id:
            raise ValueError("config_id is required")
        if self.allow_active_routing:
            raise ValueError("PROD-1 forbids active routing")
        if self.allow_payload_transfer:
            raise ValueError("PROD-1 forbids payload transfer")
        if self.allow_writes:
            raise ValueError("PROD-1 forbids writes")
        if self.allow_commit_execution:
            raise ValueError("PROD-1 forbids commit execution")
        if self.allow_raw_payload_trace:
            raise ValueError("PROD-1 forbids raw payload traces")
        if self.allow_production_activation:
            raise ValueError("PROD-1 forbids production activation")
        for name in ("require_source_matrix", "require_rollback_evidence", "require_kill_switch", "require_commit_gate_dry_run"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be true")
        return self


@dataclass(frozen=True)
class QSpinRuntimeFeatureFlagSnapshot:
    flags: FrozenSet[QSpinRuntimeFeatureFlag] = frozenset()

    def validate(self) -> "QSpinRuntimeFeatureFlagSnapshot":
        for flag in self.flags:
            if not isinstance(flag, QSpinRuntimeFeatureFlag):
                raise ValueError(f"invalid feature flag: {flag!r}")
        return self

    def unsafe_flags(self) -> FrozenSet[QSpinRuntimeFeatureFlag]:
        return self.flags


class QSpinRuntimeFeatureFlagEvaluator:
    def evaluate(self, snapshot: QSpinRuntimeFeatureFlagSnapshot) -> Tuple[QSpinShadowActivationBlockReason, ...]:
        snapshot.validate()
        reasons = []
        if QSpinRuntimeFeatureFlag.BRIDGE_ROUTING in snapshot.flags or QSpinRuntimeFeatureFlag.TOPOLOGY_ROUTING in snapshot.flags or QSpinRuntimeFeatureFlag.DEPTH_PHASE_EXECUTION in snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.ACTIVE_ROUTING_REQUESTED)
        if QSpinRuntimeFeatureFlag.PAYLOAD_TRANSFER in snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if {QSpinRuntimeFeatureFlag.SHARED_SLOT_WRITE, QSpinRuntimeFeatureFlag.EXTERNAL_MEMORY_WRITE, QSpinRuntimeFeatureFlag.QH_STORAGE_WRITE} & snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.WRITE_REQUESTED)
        if QSpinRuntimeFeatureFlag.COMMIT_EXECUTION in snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.COMMIT_REQUESTED)
        if QSpinRuntimeFeatureFlag.RAW_PAYLOAD_TRACE in snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.RAW_TRACE_REQUESTED)
        if QSpinRuntimeFeatureFlag.PRODUCTION_ACTIVATION in snapshot.flags:
            reasons.append(QSpinShadowActivationBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationRequest:
    request_id: str
    requested_mode: QSpinShadowActivationMode = QSpinShadowActivationMode.SHADOW_ONLY
    feature_flags: QSpinRuntimeFeatureFlagSnapshot = field(default_factory=QSpinRuntimeFeatureFlagSnapshot)
    source_matrix_complete: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinRuntimeShadowActivationRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        if not isinstance(self.requested_mode, QSpinShadowActivationMode):
            raise ValueError("requested_mode must be QSpinShadowActivationMode")
        self.feature_flags.validate()
        return self

    def idempotency_key(self) -> str:
        payload = self.request_id + "|" + self.requested_mode.value + "|" + ",".join(sorted(flag.value for flag in self.feature_flags.flags)) + "|" + str(self.source_matrix_complete)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationDecision:
    status: QSpinShadowActivationStatus
    allowed_shadow_only: bool
    block_reasons: Tuple[QSpinShadowActivationBlockReason, ...] = ()
    warnings: Tuple[str, ...] = ()

    def validate(self) -> "QSpinRuntimeShadowActivationDecision":
        if self.status is QSpinShadowActivationStatus.ALLOWED_SHADOW_ONLY and not self.allowed_shadow_only:
            raise ValueError("allowed status requires allowed_shadow_only")
        if self.status is QSpinShadowActivationStatus.BLOCKED and not self.block_reasons:
            raise ValueError("blocked status requires block reasons")
        return self


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationTrace:
    request_id: str
    idempotency_key: str
    status: QSpinShadowActivationStatus
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "idempotency_key": self.idempotency_key, "status": self.status.value, "safe_summary": dict(self.safe_summary)}


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationAuditEvent:
    event_id: str
    request_id: str
    action: str
    status: QSpinShadowActivationStatus
    secret_free: bool = True
    raw_payload_free: bool = True

    def validate(self) -> "QSpinRuntimeShadowActivationAuditEvent":
        if not self.event_id or not self.request_id or not self.action:
            raise ValueError("event_id, request_id, and action are required")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("audit events must be secret-free and raw-payload-free")
        return self


@dataclass(frozen=True)
class QSpinRuntimeShadowActivationResult:
    request: QSpinRuntimeShadowActivationRequest
    decision: QSpinRuntimeShadowActivationDecision
    trace: QSpinRuntimeShadowActivationTrace
    audit_event: QSpinRuntimeShadowActivationAuditEvent
    routed_live_data: bool = False
    transferred_payload: bool = False
    wrote_state: bool = False
    executed_commit: bool = False
    production_activated: bool = False

    def validate(self) -> "QSpinRuntimeShadowActivationResult":
        self.request.validate()
        self.decision.validate()
        self.audit_event.validate()
        if any([self.routed_live_data, self.transferred_payload, self.wrote_state, self.executed_commit, self.production_activated]):
            raise ValueError("PROD-1 shadow activation result must not perform live effects")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "decision": {"status": self.decision.status.value, "allowed_shadow_only": self.decision.allowed_shadow_only, "block_reasons": [r.value for r in self.decision.block_reasons], "warnings": list(self.decision.warnings)},
            "trace": self.trace.to_dict(),
            "routed_live_data": self.routed_live_data,
            "transferred_payload": self.transferred_payload,
            "wrote_state": self.wrote_state,
            "executed_commit": self.executed_commit,
            "production_activated": self.production_activated,
        }


class QSpinRuntimeShadowActivationController:
    def __init__(self, config: Optional[QSpinRuntimeShadowActivationConfig] = None, production_config: Optional[QSpinProductionConfig] = None):
        self.config = (config or build_default_qspin_shadow_activation_config()).validate()
        self.production_config = (production_config or build_default_qspin_production_config()).validate()
        self._history: Dict[str, QSpinRuntimeShadowActivationResult] = {}

    def evaluate(
        self,
        request: QSpinRuntimeShadowActivationRequest,
        *,
        kill_switch_decision: QSpinKillSwitchDecision,
        commit_gate_result: QSpinCommitGateDryRunResult,
        rollback_result: QSpinRollbackDryRunResult,
    ) -> QSpinRuntimeShadowActivationResult:
        request.validate()
        key = request.idempotency_key()
        if key in self._history:
            prior = self._history[key]
            decision = QSpinRuntimeShadowActivationDecision(QSpinShadowActivationStatus.IDEMPOTENT_REPLAY, prior.decision.allowed_shadow_only, prior.decision.block_reasons, prior.decision.warnings).validate()
            trace = QSpinRuntimeShadowActivationTrace(request.request_id, key, decision.status, {"idempotent_replay": True, "prior_status": prior.decision.status.value})
            event = QSpinRuntimeShadowActivationAuditEvent("audit_" + request.request_id, request.request_id, "idempotent_replay", decision.status).validate()
            return QSpinRuntimeShadowActivationResult(request, decision, trace, event).validate()

        reasons = list(QSpinRuntimeFeatureFlagEvaluator().evaluate(request.feature_flags))
        warnings = []
        try:
            self.production_config.validate()
        except ValueError:
            reasons.append(QSpinShadowActivationBlockReason.CONFIG_INVALID)
        if request.requested_mode is not QSpinShadowActivationMode.SHADOW_ONLY:
            reasons.append(QSpinShadowActivationBlockReason.MODE_DISABLED)
        if not request.source_matrix_complete:
            reasons.append(QSpinShadowActivationBlockReason.SOURCE_MATRIX_INCOMPLETE)
        if not kill_switch_decision.allowed_for_shadow:
            reasons.append(QSpinShadowActivationBlockReason.KILL_SWITCH_NOT_ALLOWING)
        if not commit_gate_result.decision.allowed_for_shadow:
            reasons.append(QSpinShadowActivationBlockReason.COMMIT_GATE_NOT_ALLOWING)
        if not rollback_result.passed:
            reasons.append(QSpinShadowActivationBlockReason.ROLLBACK_EVIDENCE_MISSING)

        if reasons:
            decision = QSpinRuntimeShadowActivationDecision(QSpinShadowActivationStatus.BLOCKED, False, tuple(dict.fromkeys(reasons)), tuple(warnings)).validate()
        else:
            decision = QSpinRuntimeShadowActivationDecision(QSpinShadowActivationStatus.ALLOWED_SHADOW_ONLY, True, (), ("shadow inspection only; no live routing",)).validate()
        trace = QSpinRuntimeShadowActivationTrace(request.request_id, key, decision.status, {
            "source_matrix_complete": request.source_matrix_complete,
            "kill_switch_allowed": kill_switch_decision.allowed_for_shadow,
            "commit_gate_allowed": commit_gate_result.decision.allowed_for_shadow,
            "rollback_passed": rollback_result.passed,
            "block_count": len(decision.block_reasons),
        })
        event = QSpinRuntimeShadowActivationAuditEvent("audit_" + request.request_id, request.request_id, "evaluate_shadow_activation", decision.status).validate()
        result = QSpinRuntimeShadowActivationResult(request, decision, trace, event).validate()
        self._history[key] = result
        return result


def build_default_qspin_shadow_activation_config() -> QSpinRuntimeShadowActivationConfig:
    return QSpinRuntimeShadowActivationConfig().validate()


def validate_qspin_shadow_activation_config(config: QSpinRuntimeShadowActivationConfig) -> QSpinRuntimeShadowActivationConfig:
    return config.validate()
