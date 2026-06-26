"""Experimental QSPIN live activation contracts.

This module is intentionally separate from the existing PROD-1 shadow-only
contracts. It allows opt-in experimental live effects while keeping fail-closed
kill-switch, source evidence, write-permission, and raw-payload redaction
semantics explicit in every trace.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Tuple


class QSpinExperimentalLiveMode(str, Enum):
    DISABLED = "disabled"
    EXPERIMENTAL_LIVE = "experimental_live"


class QSpinExperimentalLiveStatus(str, Enum):
    DISABLED = "disabled"
    ALLOWED_EXPERIMENTAL_LIVE = "allowed_experimental_live"
    BLOCKED = "blocked"


class QSpinExperimentalLiveBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    KILL_SWITCH_NOT_ENABLED = "kill_switch_not_enabled"
    SOURCE_MATRIX_INCOMPLETE = "source_matrix_incomplete"
    ROLLBACK_EVIDENCE_MISSING = "rollback_evidence_missing"
    WRITE_PERMISSION_MISSING = "write_permission_missing"
    RAW_PAYLOAD_TRACE_REQUESTED = "raw_payload_trace_requested"
    FEATURE_DISABLED = "feature_disabled"


@dataclass(frozen=True)
class QSpinExperimentalLiveConfig:
    enabled: bool = False
    mode: QSpinExperimentalLiveMode = QSpinExperimentalLiveMode.DISABLED
    allow_live_routing: bool = False
    allow_payload_transfer: bool = False
    allow_shared_slot_write: bool = False
    allow_qh_storage_write: bool = False
    allow_commit_execution: bool = False
    allow_raw_payload_trace: bool = False
    max_payload_tokens: int = 8
    payload_scale: float = 0.05
    routing_scale: float = 0.10

    def validate(self) -> "QSpinExperimentalLiveConfig":
        if self.enabled and self.mode is not QSpinExperimentalLiveMode.EXPERIMENTAL_LIVE:
            raise ValueError("enabled QSPIN live config requires experimental_live mode")
        if not self.enabled and self.mode is not QSpinExperimentalLiveMode.DISABLED:
            raise ValueError("disabled QSPIN live config must use disabled mode")
        if self.allow_raw_payload_trace:
            raise ValueError("raw payload traces are forbidden in experimental live mode")
        if self.max_payload_tokens <= 0:
            raise ValueError("max_payload_tokens must be positive")
        if not 0.0 <= float(self.payload_scale) <= 1.0:
            raise ValueError("payload_scale must be in [0,1]")
        if not 0.0 <= float(self.routing_scale) <= 1.0:
            raise ValueError("routing_scale must be in [0,1]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["mode"] = self.mode.value
        return out


@dataclass(frozen=True)
class QSpinExperimentalLiveRequest:
    request_id: str
    operation: str
    mode: QSpinExperimentalLiveMode
    source_matrix_complete: bool
    rollback_evidence_present: bool
    kill_switch_enabled: bool
    write_permission_present: bool = False
    live_routing_requested: bool = True
    payload_transfer_requested: bool = True
    shared_slot_write_requested: bool = False
    qh_storage_write_requested: bool = False
    commit_execution_requested: bool = False
    raw_payload_trace_requested: bool = False
    metadata: Mapping[str, Any] = None

    def validate(self) -> "QSpinExperimentalLiveRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.operation not in {"read", "process", "write"}:
            raise ValueError("operation must be read/process/write")
        if not isinstance(self.mode, QSpinExperimentalLiveMode):
            raise ValueError("mode must be QSpinExperimentalLiveMode")
        return self


@dataclass(frozen=True)
class QSpinExperimentalLiveDecision:
    status: QSpinExperimentalLiveStatus
    allowed_experimental_live: bool
    live_routing: bool = False
    payload_transfer: bool = False
    shared_slot_write: bool = False
    qh_storage_write: bool = False
    commit_execution: bool = False
    block_reasons: Tuple[QSpinExperimentalLiveBlockReason, ...] = ()
    warnings: Tuple[str, ...] = ()

    def validate(self) -> "QSpinExperimentalLiveDecision":
        if self.status is QSpinExperimentalLiveStatus.ALLOWED_EXPERIMENTAL_LIVE and not self.allowed_experimental_live:
            raise ValueError("allowed status requires allowed_experimental_live")
        if self.status is QSpinExperimentalLiveStatus.BLOCKED and not self.block_reasons:
            raise ValueError("blocked status requires block reasons")
        if not self.allowed_experimental_live and any(
            [self.live_routing, self.payload_transfer, self.shared_slot_write, self.qh_storage_write, self.commit_execution]
        ):
            raise ValueError("blocked/disabled decisions cannot enable live effects")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed_experimental_live": self.allowed_experimental_live,
            "live_routing": self.live_routing,
            "payload_transfer": self.payload_transfer,
            "shared_slot_write": self.shared_slot_write,
            "qh_storage_write": self.qh_storage_write,
            "commit_execution": self.commit_execution,
            "block_reasons": [r.value for r in self.block_reasons],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class QSpinExperimentalLiveResult:
    request: QSpinExperimentalLiveRequest
    decision: QSpinExperimentalLiveDecision
    trace: Mapping[str, Any]

    def validate(self) -> "QSpinExperimentalLiveResult":
        self.request.validate()
        self.decision.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": {
                "request_id": self.request.request_id,
                "operation": self.request.operation,
                "mode": self.request.mode.value,
                "source_matrix_complete": self.request.source_matrix_complete,
                "rollback_evidence_present": self.request.rollback_evidence_present,
                "kill_switch_enabled": self.request.kill_switch_enabled,
            },
            "decision": self.decision.to_dict(),
            "trace": dict(self.trace),
        }


class QSpinExperimentalLiveActivationController:
    def __init__(self, config: QSpinExperimentalLiveConfig | None = None):
        self.config = (config or QSpinExperimentalLiveConfig()).validate()

    def evaluate(self, request: QSpinExperimentalLiveRequest) -> QSpinExperimentalLiveResult:
        request.validate()
        reasons = []
        if not self.config.enabled or request.mode is not QSpinExperimentalLiveMode.EXPERIMENTAL_LIVE:
            reasons.append(QSpinExperimentalLiveBlockReason.MODE_DISABLED)
        if not request.kill_switch_enabled:
            reasons.append(QSpinExperimentalLiveBlockReason.KILL_SWITCH_NOT_ENABLED)
        if not request.source_matrix_complete:
            reasons.append(QSpinExperimentalLiveBlockReason.SOURCE_MATRIX_INCOMPLETE)
        if not request.rollback_evidence_present:
            reasons.append(QSpinExperimentalLiveBlockReason.ROLLBACK_EVIDENCE_MISSING)
        if request.raw_payload_trace_requested:
            reasons.append(QSpinExperimentalLiveBlockReason.RAW_PAYLOAD_TRACE_REQUESTED)

        live_routing = bool(request.live_routing_requested and self.config.allow_live_routing)
        payload_transfer = bool(request.payload_transfer_requested and self.config.allow_payload_transfer)
        shared_slot_write = bool(request.shared_slot_write_requested and self.config.allow_shared_slot_write)
        qh_storage_write = bool(request.qh_storage_write_requested and self.config.allow_qh_storage_write)
        commit_execution = bool(request.commit_execution_requested and self.config.allow_commit_execution)

        if request.live_routing_requested and not self.config.allow_live_routing:
            reasons.append(QSpinExperimentalLiveBlockReason.FEATURE_DISABLED)
        if request.payload_transfer_requested and not self.config.allow_payload_transfer:
            reasons.append(QSpinExperimentalLiveBlockReason.FEATURE_DISABLED)
        if request.shared_slot_write_requested and not self.config.allow_shared_slot_write:
            reasons.append(QSpinExperimentalLiveBlockReason.FEATURE_DISABLED)
        if request.qh_storage_write_requested and not self.config.allow_qh_storage_write:
            reasons.append(QSpinExperimentalLiveBlockReason.FEATURE_DISABLED)
        if request.commit_execution_requested and not self.config.allow_commit_execution:
            reasons.append(QSpinExperimentalLiveBlockReason.FEATURE_DISABLED)
        if (shared_slot_write or qh_storage_write or commit_execution) and not request.write_permission_present:
            reasons.append(QSpinExperimentalLiveBlockReason.WRITE_PERMISSION_MISSING)

        if reasons:
            decision = QSpinExperimentalLiveDecision(
                status=QSpinExperimentalLiveStatus.BLOCKED,
                allowed_experimental_live=False,
                block_reasons=tuple(dict.fromkeys(reasons)),
                warnings=("experimental live QSPIN blocked fail-closed",),
            )
        else:
            decision = QSpinExperimentalLiveDecision(
                status=QSpinExperimentalLiveStatus.ALLOWED_EXPERIMENTAL_LIVE,
                allowed_experimental_live=True,
                live_routing=live_routing,
                payload_transfer=payload_transfer,
                shared_slot_write=shared_slot_write,
                qh_storage_write=qh_storage_write,
                commit_execution=commit_execution,
                warnings=("experimental live mode; not production readiness",),
            )
        trace = {
            "trace_type": "qspin_experimental_live_activation",
            "raw_payload_free": True,
            "secret_free": True,
            "max_payload_tokens": int(self.config.max_payload_tokens),
            "payload_scale": float(self.config.payload_scale),
            "routing_scale": float(self.config.routing_scale),
        }
        return QSpinExperimentalLiveResult(request, decision.validate(), trace).validate()


def build_qspin_experimental_live_config(enabled: bool = False) -> QSpinExperimentalLiveConfig:
    if not enabled:
        return QSpinExperimentalLiveConfig().validate()
    return QSpinExperimentalLiveConfig(
        enabled=True,
        mode=QSpinExperimentalLiveMode.EXPERIMENTAL_LIVE,
        allow_live_routing=True,
        allow_payload_transfer=True,
        allow_shared_slot_write=True,
        allow_qh_storage_write=True,
        allow_commit_execution=True,
    ).validate()
