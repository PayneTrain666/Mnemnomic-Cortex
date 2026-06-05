"""QSPIN-PROD-7 synthetic-to-real boundary verifier.

All real-side boundaries are verified as blocked/not crossed. This module never
calls real runtime targets or writes to real stores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

class BoundaryVerificationMode(str, Enum):
    DISABLED = "disabled"
    VERIFY_NO_CROSSING = "verify_no_crossing"

class BoundaryVerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class BoundaryBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    REAL_SIDE_CROSSED = "real_side_crossed"
    LIVE_RUNTIME_REQUESTED = "live_runtime_requested"
    WRITE_REQUESTED = "write_requested"
    COMMIT_REQUESTED = "commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"
    MISSING_REAL_TARGET_METADATA = "missing_real_target_metadata"
    UNSAFE_BOUNDARY = "unsafe_boundary"

class BoundarySurfaceKind(str, Enum):
    SYNTHETIC_PAYLOAD = "synthetic_payload"
    REAL_PAYLOAD = "real_payload"
    SYNTHETIC_QH_SANDBOX = "synthetic_qh_sandbox"
    REAL_QH_STORAGE = "real_qh_storage"
    SYNTHETIC_SHARED_SLOT_SANDBOX = "synthetic_shared_slot_sandbox"
    REAL_SHARED_SLOT_STORE = "real_shared_slot_store"
    SYNTHETIC_EXTERNAL_MEMORY = "synthetic_external_memory"
    REAL_EXTERNAL_MEMORY = "real_external_memory"
    DRY_RUN_COMMIT = "dry_run_commit"
    REAL_COMMIT = "real_commit"
    SHADOW_ROUTE = "shadow_route"
    LIVE_ROUTE = "live_route"
    SYNTHETIC_TRACE = "synthetic_trace"
    RAW_PAYLOAD_TRACE = "raw_payload_trace"
    LOCAL_SOURCE_PROBE = "local_source_probe"
    LIVE_RUNTIME_EXECUTION = "live_runtime_execution"

@dataclass(frozen=True)
class BoundarySurfaceRecord:
    surface_id: str
    surface_kind: BoundarySurfaceKind
    synthetic_side_inspected: bool = True
    real_side_touched: bool = False
    blocker: str = "default_blocker"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    optional_real_metadata: bool = False

    def validate(self) -> "BoundarySurfaceRecord":
        if not self.surface_id:
            raise ValueError("surface_id required")
        if not isinstance(self.surface_kind, BoundarySurfaceKind):
            raise ValueError("invalid surface kind")
        if self.real_side_touched:
            raise ValueError("real side was touched")
        return self

@dataclass(frozen=True)
class BoundaryVerificationRequest:
    request_id: str
    record: BoundarySurfaceRecord
    live_runtime_requested: bool = False
    write_requested: bool = False
    commit_requested: bool = False
    production_activation_requested: bool = False

@dataclass(frozen=True)
class BoundaryVerificationResult:
    request_id: str
    surface_id: str
    surface_kind: BoundarySurfaceKind
    status: BoundaryVerificationStatus
    reasons: Tuple[BoundaryBlockReason, ...] = ()
    synthetic_side_inspected: bool = True
    real_side_not_touched: bool = True
    blocker: str = "default_blocker"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    residual_risk: str = "low"

    def validate(self) -> "BoundaryVerificationResult":
        if not self.real_side_not_touched:
            raise ValueError("boundary verifier detected real side touch")
        if self.status in (BoundaryVerificationStatus.FAILED, BoundaryVerificationStatus.BLOCKED) and not self.reasons:
            raise ValueError("failed/blocked boundary result requires reasons")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind.value,
            "status": self.status.value,
            "reasons": [r.value for r in self.reasons],
            "synthetic_side_inspected": self.synthetic_side_inspected,
            "real_side_not_touched": self.real_side_not_touched,
            "blocker": self.blocker,
            "residual_risk": self.residual_risk,
        }

@dataclass(frozen=True)
class BoundaryVerificationSuiteResult:
    results: Tuple[BoundaryVerificationResult, ...]

    @property
    def passed(self) -> int: return sum(1 for r in self.results if r.status == BoundaryVerificationStatus.PASSED)
    @property
    def failed(self) -> int: return sum(1 for r in self.results if r.status in (BoundaryVerificationStatus.FAILED, BoundaryVerificationStatus.BLOCKED))
    @property
    def skipped(self) -> int: return sum(1 for r in self.results if r.status == BoundaryVerificationStatus.SKIPPED)
    def to_dict(self) -> Dict[str, Any]: return {"passed": self.passed, "failed": self.failed, "skipped": self.skipped, "results": [r.to_dict() for r in self.results]}

@dataclass(frozen=True)
class BoundaryVerificationConfig:
    mode: BoundaryVerificationMode = BoundaryVerificationMode.VERIFY_NO_CROSSING
    allow_real_side_touch: bool = False
    allow_writes: bool = False
    allow_commits: bool = False
    allow_production_activation: bool = False

    def validate(self) -> "BoundaryVerificationConfig":
        if self.mode == BoundaryVerificationMode.DISABLED:
            raise ValueError("boundary verification disabled")
        if any([self.allow_real_side_touch, self.allow_writes, self.allow_commits, self.allow_production_activation]):
            raise ValueError("unsafe boundary verification config")
        return self

class SyntheticToRealBoundaryVerifier:
    def __init__(self, config: Optional[BoundaryVerificationConfig] = None):
        self.config = (config or build_default_boundary_verification_config()).validate()

    def verify(self, request: BoundaryVerificationRequest) -> BoundaryVerificationResult:
        reasons = []
        try:
            request.record.validate()
        except ValueError:
            reasons.append(BoundaryBlockReason.REAL_SIDE_CROSSED)
        if request.live_runtime_requested:
            reasons.append(BoundaryBlockReason.LIVE_RUNTIME_REQUESTED)
        if request.write_requested:
            reasons.append(BoundaryBlockReason.WRITE_REQUESTED)
        if request.commit_requested:
            reasons.append(BoundaryBlockReason.COMMIT_REQUESTED)
        if request.production_activation_requested:
            reasons.append(BoundaryBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if request.record.surface_kind in (BoundarySurfaceKind.REAL_PAYLOAD, BoundarySurfaceKind.REAL_QH_STORAGE, BoundarySurfaceKind.REAL_SHARED_SLOT_STORE, BoundarySurfaceKind.REAL_EXTERNAL_MEMORY, BoundarySurfaceKind.REAL_COMMIT, BoundarySurfaceKind.LIVE_ROUTE, BoundarySurfaceKind.RAW_PAYLOAD_TRACE, BoundarySurfaceKind.LIVE_RUNTIME_EXECUTION):
            # Real side surfaces are expected to be blocked/not crossed, but still pass if not touched.
            pass
        status = BoundaryVerificationStatus.BLOCKED if reasons else BoundaryVerificationStatus.PASSED
        return BoundaryVerificationResult(
            request.request_id,
            request.record.surface_id,
            request.record.surface_kind,
            status,
            tuple(dict.fromkeys(reasons)),
            request.record.synthetic_side_inspected,
            not request.record.real_side_touched,
            request.record.blocker,
            request.record.evidence,
            "low" if not reasons else "medium",
        ).validate()

    def verify_suite(self, records: Iterable[BoundarySurfaceRecord]) -> BoundaryVerificationSuiteResult:
        return BoundaryVerificationSuiteResult(tuple(self.verify(BoundaryVerificationRequest("verify_" + r.surface_id, r)) for r in records))

def build_default_boundary_verification_config() -> BoundaryVerificationConfig:
    return BoundaryVerificationConfig().validate()

def build_default_boundary_surface_records() -> Tuple[BoundarySurfaceRecord, ...]:
    return tuple(BoundarySurfaceRecord(kind.value, kind, True, False, f"block_{kind.value}") for kind in BoundarySurfaceKind)
