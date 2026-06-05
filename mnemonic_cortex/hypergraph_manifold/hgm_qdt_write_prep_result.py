"""HGM/QDT write-preparation result contracts.

This module is read-only by design.  It defines preview/contract records that
explain how HGM bridge outputs could become QDT/WM write proposals in a later
permissioned stage.  No class here executes a QDT/WM write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import hashlib

from .enums import DepthLayer, GeometryType, TraceEventKind, ValidationSeverity
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def write_prep_stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def write_prep_redact(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): write_prep_redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(write_prep_redact(key, v) for v in value)
    return value


def trace_write_prep(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): write_prep_redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


@dataclass(frozen=True)
class HGMQDTWritePrepOptions:
    proposal_dim: int = 16
    max_payloads: int = 128
    max_hooks: int = 128
    max_probe_symbols: int = 128
    default_memory_type: str = "wm"
    default_bank_name: str = "qdt_working_memory"
    default_task_mode: str = "quantum_holographic"
    default_triplet_index: int = 0
    allow_torch_probe: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.proposal_dim) <= 0:
            raise ValueError("proposal_dim must be positive")
        if int(self.max_payloads) <= 0:
            raise ValueError("max_payloads must be positive")
        if int(self.max_hooks) <= 0:
            raise ValueError("max_hooks must be positive")
        if int(self.max_probe_symbols) <= 0:
            raise ValueError("max_probe_symbols must be positive")
        if int(self.default_triplet_index) not in (0, 1, 2):
            raise ValueError("default_triplet_index must be 0, 1, or 2")
        object.__setattr__(self, "proposal_dim", int(self.proposal_dim))
        object.__setattr__(self, "max_payloads", int(self.max_payloads))
        object.__setattr__(self, "max_hooks", int(self.max_hooks))
        object.__setattr__(self, "max_probe_symbols", int(self.max_probe_symbols))
        object.__setattr__(self, "default_triplet_index", int(self.default_triplet_index))


@dataclass(frozen=True)
class QDTContractSymbolProbe:
    symbol_name: str
    module_path: str
    present: bool
    signature: str
    required_fields: Tuple[str, ...]
    missing_fields: Tuple[str, ...]
    compatible: bool
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", tuple(self.required_fields or tuple()))
        object.__setattr__(self, "missing_fields", tuple(self.missing_fields or tuple()))


@dataclass(frozen=True)
class QDTWriteContractProbeResult:
    probes: Tuple[QDTContractSymbolProbe, ...]
    compatible: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probes", tuple(self.probes or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class TensorProposalPreview:
    preview_id: str
    source_payload_id: str
    content_vector: Tuple[float, ...]
    content_shape: Tuple[int, ...]
    content_fingerprint: str
    finite: bool
    bounded: bool
    write_permission: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_vector", tuple(float(v) for v in self.content_vector))
        object.__setattr__(self, "content_shape", tuple(int(v) for v in self.content_shape))


@dataclass(frozen=True)
class ProposalMaterializationContract:
    contract_id: str
    tensor_previews: Tuple[TensorProposalPreview, ...]
    proposal_dim: int
    proposal_class_name: str
    write_permission_default: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensor_previews", tuple(self.tensor_previews or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class SlotIDMappingRecord:
    mapping_id: str
    source_hook_id: str
    hgm_target_slot_id: str
    wm_local_slot_id: str
    wm_canonical_slot_id: str
    namespace: str
    content_fingerprint: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotIDMappingPlan:
    plan_id: str
    mappings: Tuple[SlotIDMappingRecord, ...]
    namespace: str
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mappings", tuple(self.mappings or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class QSpinQHConversionRecord:
    conversion_id: str
    qspin_signature_id: str
    depth_index: int
    geometry_map: str
    memory_type: str
    task_mode: str
    bank_name: str
    triplet_index: int
    qh_composite_code: str
    qh_record_id_preview: str
    schema_preview: Mapping[str, Any]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QSpinQHConversionContract:
    contract_id: str
    conversions: Tuple[QSpinQHConversionRecord, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conversions", tuple(self.conversions or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RollbackSnapshotRequirement:
    requirement_id: str
    operation_id: str
    target_slot_id: str
    snapshot_source: str
    snapshot_ref_preview: str
    bound_to_actual_snapshot: bool
    required: bool
    blocking_reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackSnapshotHandshake:
    handshake_id: str
    requirements: Tuple[RollbackSnapshotRequirement, ...]
    complete: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", tuple(self.requirements or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGMQDTWritePrepResult:
    contract_probe: QDTWriteContractProbeResult
    proposal_contract: ProposalMaterializationContract
    slot_mapping_plan: SlotIDMappingPlan
    qspin_qh_contract: QSpinQHConversionContract
    rollback_handshake: RollbackSnapshotHandshake
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
