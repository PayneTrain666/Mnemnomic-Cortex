"""QSPIN-PROD-7 expanded observability review."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

class ObservabilityReviewMode(str, Enum):
    DISABLED = "disabled"
    METADATA_ONLY = "metadata_only"

class ObservabilityReviewStatus(str, Enum):
    PASSED = "passed"
    GAP_FOUND = "gap_found"
    BLOCKED = "blocked"

class ObservabilityReviewBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    RAW_PAYLOAD_PRESENT = "raw_payload_present"
    SECRET_PRESENT = "secret_present"
    MISSING_SIGNAL = "missing_signal"

class ObservabilitySignalKind(str, Enum):
    PROD1_AUDIT = "prod1_audit"
    PROD2_METRIC = "prod2_metric"
    PROD3_OBSERVABILITY = "prod3_observability"
    PROD4_OBSERVABILITY = "prod4_observability"
    PROD5_OBSERVABILITY = "prod5_observability"
    PROD6_OBSERVABILITY = "prod6_observability"
    KILL_SWITCH = "kill_switch"
    COMMIT_GATE = "commit_gate"
    ROLLBACK = "rollback"
    DEAD_LETTER = "dead_letter"
    RAW_PAYLOAD_REDACTION = "raw_payload_redaction"
    SECRET_REDACTION = "secret_redaction"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    CI_MATRIX = "ci_matrix"
    SAFETY_REGRESSION = "safety_regression"
    REMEDIATION_CLOSURE = "remediation_closure"

@dataclass(frozen=True)
class ObservabilitySignalRecord:
    signal_id: str
    signal_kind: ObservabilitySignalKind
    present: bool = True
    contains_raw_payload: bool = False
    contains_secret: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ObservabilityReviewRequest:
    request_id: str
    signals: Tuple[ObservabilitySignalRecord, ...]

@dataclass(frozen=True)
class ObservabilityReviewResult:
    request_id: str
    status: ObservabilityReviewStatus
    reasons: Tuple[ObservabilityReviewBlockReason, ...] = ()
    gap_list: Tuple[str, ...] = ()
    remediation_hints: Tuple[str, ...] = ()
    prod8_carry_forward: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "status": self.status.value, "reasons": [r.value for r in self.reasons], "gaps": list(self.gap_list), "remediation_hints": list(self.remediation_hints), "prod8_carry_forward": list(self.prod8_carry_forward)}

@dataclass(frozen=True)
class ObservabilityReviewSuiteResult:
    result: ObservabilityReviewResult

@dataclass(frozen=True)
class ObservabilityReviewConfig:
    mode: ObservabilityReviewMode = ObservabilityReviewMode.METADATA_ONLY
    require_all_signals: bool = True

    def validate(self) -> "ObservabilityReviewConfig":
        if self.mode == ObservabilityReviewMode.DISABLED:
            raise ValueError("observability review disabled")
        return self

class ObservabilityReviewEngine:
    def __init__(self, config: Optional[ObservabilityReviewConfig] = None):
        self.config = (config or build_default_observability_review_config()).validate()

    def review(self, request: ObservabilityReviewRequest) -> ObservabilityReviewResult:
        reasons = []
        gaps = []
        for s in request.signals:
            if s.contains_raw_payload:
                reasons.append(ObservabilityReviewBlockReason.RAW_PAYLOAD_PRESENT)
            if s.contains_secret:
                reasons.append(ObservabilityReviewBlockReason.SECRET_PRESENT)
            if self.config.require_all_signals and not s.present:
                reasons.append(ObservabilityReviewBlockReason.MISSING_SIGNAL)
                gaps.append(s.signal_id)
        status = ObservabilityReviewStatus.BLOCKED if any(r in reasons for r in (ObservabilityReviewBlockReason.RAW_PAYLOAD_PRESENT, ObservabilityReviewBlockReason.SECRET_PRESENT)) else (ObservabilityReviewStatus.GAP_FOUND if gaps else ObservabilityReviewStatus.PASSED)
        hints = tuple(f"restore signal {g}" for g in gaps)
        return ObservabilityReviewResult(request.request_id, status, tuple(dict.fromkeys(reasons)), tuple(gaps), hints, hints)

def build_default_observability_review_config() -> ObservabilityReviewConfig:
    return ObservabilityReviewConfig().validate()

def build_default_observability_signal_records() -> Tuple[ObservabilitySignalRecord, ...]:
    return tuple(ObservabilitySignalRecord(k.value, k, True) for k in ObservabilitySignalKind)
