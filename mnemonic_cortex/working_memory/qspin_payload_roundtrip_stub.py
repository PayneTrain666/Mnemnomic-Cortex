"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-3 metadata-safe payload codec roundtrip stubs.

Roundtrips metadata only. No raw payloads, tensors, payload transfer, storage,
shared-slot writes, external-memory writes, or QH writes are permitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple
import hashlib
import json


class QSpinPayloadRoundtripMode(str, Enum):
    DISABLED = "disabled"
    METADATA_STUB_ONLY = "metadata_stub_only"


class QSpinPayloadRoundtripStatus(str, Enum):
    ROUNDTRIP_STUB_OK = "roundtrip_stub_ok"
    BLOCKED = "blocked"


class QSpinPayloadRoundtripBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    RAW_PAYLOAD_PRESENT = "raw_payload_present"
    RAW_TENSOR_PRESENT = "raw_tensor_present"
    UNSAFE_SHAPE = "unsafe_shape"
    BUDGET_EXCEEDED = "budget_exceeded"
    NORM_BAND_INVALID = "norm_band_invalid"
    MISSING_SOURCE_OR_TARGET = "missing_source_or_target"
    WRITE_REQUESTED = "write_requested"
    TRANSFER_REQUESTED = "transfer_requested"


class QSpinPayloadCodecStubKind(str, Enum):
    DENSE = "dense"
    CHRR = "chrr"
    QH = "qh"


@dataclass(frozen=True)
class QSpinPayloadRoundtripStubPolicy:
    mode: QSpinPayloadRoundtripMode = QSpinPayloadRoundtripMode.METADATA_STUB_ONLY
    allow_raw_payload: bool = False
    allow_raw_tensor: bool = False
    allow_transfer: bool = False
    allow_storage: bool = False
    allow_writes: bool = False
    max_rank: int = 4
    max_elements: int = 1_000_000
    max_bytes: int = 16_777_216

    def validate(self) -> "QSpinPayloadRoundtripStubPolicy":
        if self.mode is QSpinPayloadRoundtripMode.DISABLED:
            raise ValueError("payload roundtrip stub disabled")
        if any([self.allow_raw_payload, self.allow_raw_tensor, self.allow_transfer, self.allow_storage, self.allow_writes]):
            raise ValueError("unsafe payload roundtrip policy")
        if self.max_rank <= 0 or self.max_elements <= 0 or self.max_bytes <= 0:
            raise ValueError("invalid payload policy limits")
        return self


@dataclass(frozen=True)
class QSpinPayloadRoundtripRequest:
    request_id: str
    stub_kind: QSpinPayloadCodecStubKind
    source_id: str
    target_id: str
    declared_shape: Tuple[int, ...]
    declared_dtype: str = "float32"
    declared_bytes: int = 0
    norm_band: Tuple[float, float] = (0.0, 1.0)
    synthetic_placeholder_id: str = "synthetic_payload_placeholder"
    raw_payload: Any = None
    raw_tensor: Any = None
    transfer_requested: bool = False
    write_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, policy: Optional[QSpinPayloadRoundtripStubPolicy] = None) -> "QSpinPayloadRoundtripRequest":
        policy = (policy or build_default_qspin_payload_roundtrip_stub_policy()).validate()
        if not self.request_id or not self.source_id or not self.target_id:
            raise ValueError("request_id, source_id, and target_id are required")
        if not isinstance(self.stub_kind, QSpinPayloadCodecStubKind):
            raise ValueError("invalid stub kind")
        if self.raw_payload is not None:
            raise ValueError("raw payload rejected")
        if self.raw_tensor is not None:
            raise ValueError("raw tensor rejected")
        if self.transfer_requested:
            raise ValueError("payload transfer rejected")
        if self.write_requested:
            raise ValueError("payload writes rejected")
        if not self.declared_shape or len(self.declared_shape) > policy.max_rank:
            raise ValueError("unsafe shape")
        n = 1
        for dim in self.declared_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("unsafe shape")
            n *= dim
        if n > policy.max_elements:
            raise ValueError("unsafe shape")
        if self.declared_bytes < 0 or self.declared_bytes > policy.max_bytes:
            raise ValueError("budget exceeded")
        lo, hi = self.norm_band
        if lo < 0 or hi < lo or hi > 1e6:
            raise ValueError("norm band invalid")
        return self

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "stub_kind": self.stub_kind.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "declared_shape": list(self.declared_shape),
            "declared_dtype": self.declared_dtype,
            "declared_bytes": self.declared_bytes,
            "norm_band": list(self.norm_band),
            "synthetic_placeholder_id": self.synthetic_placeholder_id,
        }

    def roundtrip_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.metadata_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class QSpinPayloadRoundtripDecision:
    status: QSpinPayloadRoundtripStatus
    approved: bool
    block_reasons: Tuple[QSpinPayloadRoundtripBlockReason, ...] = ()

    def validate(self) -> "QSpinPayloadRoundtripDecision":
        if self.status is QSpinPayloadRoundtripStatus.ROUNDTRIP_STUB_OK and (not self.approved or self.block_reasons):
            raise ValueError("roundtrip approval malformed")
        if self.status is QSpinPayloadRoundtripStatus.BLOCKED and (self.approved or not self.block_reasons):
            raise ValueError("blocked roundtrip decision malformed")
        return self


@dataclass(frozen=True)
class QSpinPayloadRoundtripTrace:
    request_id: str
    safe_summary: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "safe_summary": dict(self.safe_summary)}


@dataclass(frozen=True)
class QSpinPayloadRoundtripResult:
    request: QSpinPayloadRoundtripRequest
    decision: QSpinPayloadRoundtripDecision
    trace: QSpinPayloadRoundtripTrace
    metadata_in_hash: str = ""
    metadata_out_hash: str = ""
    transferred_payload: bool = False
    stored_payload: bool = False
    wrote_state: bool = False

    def validate(self) -> "QSpinPayloadRoundtripResult":
        self.decision.validate()
        if any([self.transferred_payload, self.stored_payload, self.wrote_state]):
            raise ValueError("payload roundtrip stub must not transfer/store/write")
        if self.decision.approved and self.metadata_in_hash != self.metadata_out_hash:
            raise ValueError("metadata roundtrip hash mismatch")
        return self


class QSpinPayloadCodecRoundtripStub:
    def __init__(self, policy: Optional[QSpinPayloadRoundtripStubPolicy] = None):
        self.policy = (policy or build_default_qspin_payload_roundtrip_stub_policy()).validate()

    def roundtrip(self, request: QSpinPayloadRoundtripRequest) -> QSpinPayloadRoundtripResult:
        reasons = []
        try:
            request.validate(self.policy)
        except ValueError as exc:
            msg = str(exc)
            if "raw payload" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.RAW_PAYLOAD_PRESENT)
            elif "raw tensor" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.RAW_TENSOR_PRESENT)
            elif "transfer" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.TRANSFER_REQUESTED)
            elif "write" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.WRITE_REQUESTED)
            elif "shape" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.UNSAFE_SHAPE)
            elif "budget" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.BUDGET_EXCEEDED)
            elif "norm" in msg:
                reasons.append(QSpinPayloadRoundtripBlockReason.NORM_BAND_INVALID)
            else:
                reasons.append(QSpinPayloadRoundtripBlockReason.MISSING_SOURCE_OR_TARGET)

        if reasons:
            decision = QSpinPayloadRoundtripDecision(
                QSpinPayloadRoundtripStatus.BLOCKED,
                False,
                tuple(dict.fromkeys(reasons)),
            ).validate()
            trace = QSpinPayloadRoundtripTrace(request.request_id, {"block_count": len(decision.block_reasons)})
            return QSpinPayloadRoundtripResult(request, decision, trace).validate()

        h = request.roundtrip_hash()
        decision = QSpinPayloadRoundtripDecision(QSpinPayloadRoundtripStatus.ROUNDTRIP_STUB_OK, True).validate()
        summary = request.metadata_dict()
        summary["metadata_roundtrip_hash"] = h
        trace = QSpinPayloadRoundtripTrace(request.request_id, summary)
        return QSpinPayloadRoundtripResult(request, decision, trace, h, h).validate()


def build_default_qspin_payload_roundtrip_stub_policy() -> QSpinPayloadRoundtripStubPolicy:
    return QSpinPayloadRoundtripStubPolicy().validate()
