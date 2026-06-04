"""QSPIN-PROD-1 kill-switch enforcement.

Fail-closed, auditable kill-switch primitives for shadow-only activation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class QSpinKillSwitchState(str, Enum):
    ENABLED = "enabled"
    TRIPPED = "tripped"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class QSpinKillSwitchReason(str, Enum):
    DEFAULT_SAFE = "default_safe"
    MANUAL_TRIP = "manual_trip"
    UNSAFE_CONFIG = "unsafe_config"
    SOURCE_MATRIX_FAILURE = "source_matrix_failure"
    COMMIT_GATE_BLOCK = "commit_gate_block"
    ROLLBACK_EVIDENCE_MISSING = "rollback_evidence_missing"
    DRY_RUN_RESET_APPROVED = "dry_run_reset_approved"
    UNKNOWN_STATE = "unknown_state"


@dataclass(frozen=True)
class QSpinKillSwitchDecision:
    allowed_for_shadow: bool
    state: QSpinKillSwitchState
    reasons: Tuple[QSpinKillSwitchReason, ...]
    fail_closed: bool = True

    def validate(self) -> "QSpinKillSwitchDecision":
        if self.state in {QSpinKillSwitchState.TRIPPED, QSpinKillSwitchState.DISABLED, QSpinKillSwitchState.UNKNOWN} and self.allowed_for_shadow:
            raise ValueError("unsafe kill-switch state cannot allow shadow activation")
        if not self.reasons:
            raise ValueError("kill-switch decision requires reasons")
        if not self.fail_closed:
            raise ValueError("kill-switch decisions must fail closed")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed_for_shadow": self.allowed_for_shadow, "state": self.state.value, "reasons": [r.value for r in self.reasons], "fail_closed": self.fail_closed}


@dataclass(frozen=True)
class QSpinKillSwitchAuditEvent:
    event_id: str
    action: str
    state_before: QSpinKillSwitchState
    state_after: QSpinKillSwitchState
    reasons: Tuple[QSpinKillSwitchReason, ...]
    secret_free: bool = True
    raw_payload_free: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinKillSwitchAuditEvent":
        if not self.event_id or not self.action:
            raise ValueError("event_id and action are required")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("audit events must not contain secrets or raw payloads")
        return self


@dataclass(frozen=True)
class QSpinKillSwitchTripResult:
    state: QSpinKillSwitchState
    audit_event: QSpinKillSwitchAuditEvent

    def validate(self) -> "QSpinKillSwitchTripResult":
        if self.state is not QSpinKillSwitchState.TRIPPED:
            raise ValueError("trip result must end in TRIPPED state")
        self.audit_event.validate()
        return self


@dataclass(frozen=True)
class QSpinKillSwitchResetRequest:
    request_id: str
    dry_run_approval_present: bool
    reset_reason: QSpinKillSwitchReason = QSpinKillSwitchReason.DRY_RUN_RESET_APPROVED

    def validate(self) -> "QSpinKillSwitchResetRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.dry_run_approval_present:
            raise ValueError("kill-switch reset requires explicit dry-run approval")
        return self


@dataclass(frozen=True)
class QSpinKillSwitchResetDecision:
    reset_allowed: bool
    state_after: QSpinKillSwitchState
    reasons: Tuple[QSpinKillSwitchReason, ...]
    runtime_activated: bool = False

    def validate(self) -> "QSpinKillSwitchResetDecision":
        if self.runtime_activated:
            raise ValueError("kill-switch reset must not activate runtime")
        if self.reset_allowed and self.state_after is not QSpinKillSwitchState.ENABLED:
            raise ValueError("allowed reset must end in ENABLED state")
        return self


@dataclass
class QSpinKillSwitch:
    state: QSpinKillSwitchState = QSpinKillSwitchState.ENABLED
    audit_events: Tuple[QSpinKillSwitchAuditEvent, ...] = ()

    def validate(self) -> "QSpinKillSwitch":
        if self.state is QSpinKillSwitchState.DISABLED:
            raise ValueError("kill switch must never be disabled")
        if self.state is QSpinKillSwitchState.UNKNOWN:
            raise ValueError("unknown kill-switch state fails closed")
        return self

    def decision(self) -> QSpinKillSwitchDecision:
        if self.state is QSpinKillSwitchState.ENABLED:
            return QSpinKillSwitchDecision(True, self.state, (QSpinKillSwitchReason.DEFAULT_SAFE,)).validate()
        if self.state is QSpinKillSwitchState.TRIPPED:
            return QSpinKillSwitchDecision(False, self.state, (QSpinKillSwitchReason.MANUAL_TRIP,)).validate()
        return QSpinKillSwitchDecision(False, self.state, (QSpinKillSwitchReason.UNKNOWN_STATE,)).validate()

    def trip(self, reason: QSpinKillSwitchReason = QSpinKillSwitchReason.MANUAL_TRIP, event_id: str = "qspin_kill_switch_trip") -> QSpinKillSwitchTripResult:
        before = self.state
        self.state = QSpinKillSwitchState.TRIPPED
        event = QSpinKillSwitchAuditEvent(event_id, "trip", before, self.state, (reason,)).validate()
        self.audit_events = self.audit_events + (event,)
        return QSpinKillSwitchTripResult(self.state, event).validate()

    def reset(self, request: QSpinKillSwitchResetRequest) -> QSpinKillSwitchResetDecision:
        request.validate()
        before = self.state
        self.state = QSpinKillSwitchState.ENABLED
        event = QSpinKillSwitchAuditEvent(request.request_id, "reset", before, self.state, (request.reset_reason,)).validate()
        self.audit_events = self.audit_events + (event,)
        return QSpinKillSwitchResetDecision(True, self.state, (request.reset_reason,)).validate()


def build_default_qspin_kill_switch() -> QSpinKillSwitch:
    return QSpinKillSwitch().validate()
