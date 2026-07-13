"""
Plain-language summary
----------------------
What this file is for: WM-side quantum holographic storage interface.
How it fits in the system: Metadata/compatible QH hooks for working memory writes.
Status: INCOMPLETE / interface-compatible
Important notes for non-coders: Persistent QH backend is still deferred per readiness docs.
"""

from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import uuid

import torch
import torch.nn.functional as F

from .wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig, tensor_fingerprint


GEOMETRY_CODEBOOK = {
    "euclidean": "geo-euc",
    "hyperbolic": "geo-hyp",
    "poincare": "geo-poincare",
    "spherical": "geo-sph",
    "torus": "geo-torus",
    "complex": "geo-complex",
    "complex_projective": "geo-cp",
    "cp_kahler": "geo-cpk",
    "subspace": "geo-subspace",
    "grassmann": "geo-grassmann",
    "spatial_se3": "geo-se3",
    "quaternion": "geo-quat",
    "dual_quaternion": "geo-dquat",
    "spcp": "geo-spcp",
    "holographic_phase": "geo-hphase",
    "product": "geo-product",
    "fiber_bundle": "geo-fiber",
    "tangent_bridge": "geo-tangent",
}

MEMORY_TYPE_CODEBOOK = {
    "wm": "mem-wm",
    "ltm": "mem-ltm",
    "mann": "mem-mann",
    "spcp": "mem-spcp",
    "shared": "mem-shared",
}

TASK_MODE_CODEBOOK = {
    "literal": "task-literal",
    "hierarchical": "task-hierarchy",
    "temporal": "task-time",
    "spatial_mechanical": "task-spatial-mech",
    "symbolic_mathematical": "task-symbolic-math",
    "procedural": "task-procedure",
    "conflict_verification": "task-conflict-verify",
    "creative_synthesis": "task-creative",
    "policy_governance": "task-policy",
    "quantum_holographic": "task-qh",
    "default": "task-default",
}


def _stable_code(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class QHCodeSchema:
    """Quantum-holographic-compatible code schema.

    This is coding metadata for future quantum/holographic-style memory
    backends. It does not claim quantum hardware behavior.

    Required carryover codes:
    - depth_code
    - bank_code
    - geometry_code
    - triplet_code
    - memory_type_code
    - task_mode_code
    """

    depth_code: str
    bank_code: str
    geometry_code: str
    triplet_code: str
    memory_type_code: str
    task_mode_code: str

    def to_tuple(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.depth_code,
            self.bank_code,
            self.geometry_code,
            self.triplet_code,
            self.memory_type_code,
            self.task_mode_code,
        )

    def composite_code(self) -> str:
        return _stable_code("qh", "|".join(self.to_tuple()))

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["composite_code"] = self.composite_code()
        out["notice"] = "quantum_holographic_compatible_metadata_not_quantum_hardware_claim"
        return out


def build_qh_code_schema(
    *,
    depth_index: int,
    bank_name: str,
    geometry_name: str,
    triplet_index: int,
    memory_type: str,
    task_mode: str = "default",
    num_depths: int = 8,
) -> QHCodeSchema:
    if not 0 <= depth_index < num_depths:
        raise ValueError(f"depth_index must be in [0,{num_depths})")
    if triplet_index not in (0, 1, 2):
        raise ValueError("triplet_index must be 0(anchor), 1(direction), or 2(phase)")
    if not bank_name:
        raise ValueError("bank_name must be non-empty")

    geometry_code = GEOMETRY_CODEBOOK.get(geometry_name, _stable_code("geo", geometry_name))
    memory_type_code = MEMORY_TYPE_CODEBOOK.get(memory_type, _stable_code("mem", memory_type))
    task_mode_code = TASK_MODE_CODEBOOK.get(task_mode, _stable_code("task", task_mode))
    return QHCodeSchema(
        depth_code=f"depth-{depth_index:02d}",
        bank_code=_stable_code("bank", bank_name),
        geometry_code=geometry_code,
        triplet_code=["triplet-anchor", "triplet-direction", "triplet-phase"][triplet_index],
        memory_type_code=memory_type_code,
        task_mode_code=task_mode_code,
    )


@dataclass
class QHInterferenceReport:
    """Interference report for QH-compatible storage records."""

    record_id: str
    compared_records: int
    max_similarity: float
    interference_detected: bool
    threshold: float
    conflicting_record_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QHStorageRecord:
    """Storage record linked to a canonical shared slot."""

    record_id: str
    canonical_slot_id: str
    code_schema: QHCodeSchema
    vector_fingerprint: str
    vector_norm: float
    confidence: float = 1.0
    write_permission_required: bool = True
    write_permission_granted: bool = False
    interference: Optional[QHInterferenceReport] = None
    created_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.record_id.startswith("qhrec-"):
            raise ValueError("record_id must start with qhrec-")
        if not self.canonical_slot_id.startswith("css-"):
            raise ValueError("canonical_slot_id must start with css-")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.vector_norm < 0.0:
            raise ValueError("vector_norm must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "canonical_slot_id": self.canonical_slot_id,
            "code_schema": self.code_schema.to_dict(),
            "vector_fingerprint": self.vector_fingerprint,
            "vector_norm": self.vector_norm,
            "confidence": self.confidence,
            "write_permission_required": self.write_permission_required,
            "write_permission_granted": self.write_permission_granted,
            "interference": None if self.interference is None else self.interference.to_dict(),
            "created_at": self.created_at,
            "metadata": self.metadata,
            "paamax_metadata": {
                "trace_type": "qh_storage_record",
                "write_permission_required": self.write_permission_required,
                "write_permission_granted": self.write_permission_granted,
                "interference_detected": False if self.interference is None else self.interference.interference_detected,
                "canonical_slot_id": self.canonical_slot_id,
                "qh_composite_code": self.code_schema.composite_code(),
            },
        }


@dataclass
class QuantumHolographicStorageConfig:
    dim: int
    num_depths: int = 8
    interference_threshold: float = 0.985
    require_write_permission: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if not 0.0 <= self.interference_threshold <= 1.0:
            raise ValueError("interference_threshold must be in [0,1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class QuantumHolographicStorage:
    """Quantum-holographic-compatible depth-coded storage interface.

    WM-4C scope:
    - Provides explicit QH-compatible code metadata.
    - Links QH records to canonical shared slots.
    - Performs vector interference checks.
    - Emits PAAMA-X write-permission/trace metadata.

    It intentionally does not claim real quantum computation or holographic
    hardware storage. It is a future-backend-compatible interface.
    """

    def __init__(self, config: QuantumHolographicStorageConfig, shared_slot_store: Optional[SharedSlotStore] = None):
        config.validate()
        self.config = config
        self.shared_slot_store = shared_slot_store or SharedSlotStore(SharedSlotStoreConfig(namespace="qh_store", dim=config.dim))
        self.records: Dict[str, QHStorageRecord] = {}
        self.vectors: Dict[str, torch.Tensor] = {}

    def _new_id(self) -> str:
        return f"qhrec-{uuid.uuid4().hex}"

    def _validate_vector(self, vector: torch.Tensor) -> None:
        if vector.dim() != 1 or vector.size(0) != self.config.dim:
            raise ValueError(f"vector must be [D={self.config.dim}]")
        if not torch.isfinite(vector).all():
            raise ValueError("vector contains NaN or Inf")

    def check_interference(self, record_id: str, vector: torch.Tensor) -> QHInterferenceReport:
        self._validate_vector(vector)
        if not self.records:
            return QHInterferenceReport(
                record_id=record_id,
                compared_records=0,
                max_similarity=0.0,
                interference_detected=False,
                threshold=self.config.interference_threshold,
            )

        v = F.normalize(vector.detach().float(), dim=0, eps=self.config.eps)
        max_similarity = 0.0
        conflicts: List[str] = []
        for other_id, other in self.vectors.items():
            if other_id == record_id:
                continue
            ov = F.normalize(other.detach().float(), dim=0, eps=self.config.eps)
            sim = float(torch.abs(torch.dot(v, ov)).detach().cpu())
            max_similarity = max(max_similarity, sim)
            if sim >= self.config.interference_threshold:
                conflicts.append(other_id)
        return QHInterferenceReport(
            record_id=record_id,
            compared_records=max(0, len(self.vectors) - (1 if record_id in self.vectors else 0)),
            max_similarity=max_similarity,
            interference_detected=bool(conflicts),
            threshold=self.config.interference_threshold,
            conflicting_record_ids=conflicts,
        )

    def create_record(
        self,
        *,
        canonical_slot_id: str,
        vector: torch.Tensor,
        depth_index: int,
        bank_name: str,
        geometry_name: str,
        triplet_index: int,
        memory_type: str,
        task_mode: str = "default",
        confidence: float = 1.0,
        write_permission: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QHStorageRecord:
        self._validate_vector(vector)
        if self.config.require_write_permission and not bool(write_permission):
            raise PermissionError("QH storage requires write_permission=True")
        if self.shared_slot_store.get_content(canonical_slot_id) is None:
            raise KeyError(f"shared slot has no stored content: {canonical_slot_id}")
        schema = build_qh_code_schema(
            depth_index=depth_index,
            bank_name=bank_name,
            geometry_name=geometry_name,
            triplet_index=triplet_index,
            memory_type=memory_type,
            task_mode=task_mode,
            num_depths=self.config.num_depths,
        )
        record_id = self._new_id()
        interference = self.check_interference(record_id, vector)
        record = QHStorageRecord(
            record_id=record_id,
            canonical_slot_id=canonical_slot_id,
            code_schema=schema,
            vector_fingerprint=tensor_fingerprint(vector),
            vector_norm=float(vector.detach().float().norm().cpu()),
            confidence=float(confidence),
            write_permission_required=self.config.require_write_permission,
            write_permission_granted=bool(write_permission) if self.config.require_write_permission else True,
            interference=interference,
            metadata=metadata or {},
        )
        record.validate()
        self.records[record_id] = record
        self.vectors[record_id] = vector.detach().clone()

        # Attach QH reference to shared slot store when the store supports it.
        attach = getattr(self.shared_slot_store, "attach_qh_record", None)
        if callable(attach):
            attach(canonical_slot_id, record_id, schema.composite_code(), interference.interference_detected)
        return record

    def create_from_shared_slot(
        self,
        *,
        canonical_slot_id: str,
        depth_index: int,
        bank_name: str,
        geometry_name: str,
        triplet_index: int,
        memory_type: str,
        task_mode: str = "default",
        confidence: float = 1.0,
        write_permission: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QHStorageRecord:
        vector = self.shared_slot_store.get_content(canonical_slot_id)
        if vector is None:
            raise KeyError(f"shared slot has no stored content: {canonical_slot_id}")
        return self.create_record(
            canonical_slot_id=canonical_slot_id,
            vector=vector,
            depth_index=depth_index,
            bank_name=bank_name,
            geometry_name=geometry_name,
            triplet_index=triplet_index,
            memory_type=memory_type,
            task_mode=task_mode,
            confidence=confidence,
            write_permission=write_permission,
            metadata=metadata,
        )

    def trace_summary(self) -> Dict[str, Any]:
        return {
            "trace_type": "quantum_holographic_storage",
            "notice": "metadata_interface_only_no_quantum_hardware_claim",
            "record_count": len(self.records),
            "record_ids": sorted(self.records.keys()),
            "paamax_metadata": {
                "trace_type": "quantum_holographic_storage",
                "write_permission_required": self.config.require_write_permission,
                "interference_threshold": self.config.interference_threshold,
                "interference_records": [
                    rid for rid, rec in self.records.items()
                    if rec.interference is not None and rec.interference.interference_detected
                ],
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "records": {rid: record.to_dict() for rid, record in self.records.items()},
            "trace": self.trace_summary(),
        }


# ---------------------------------------------------------------------------
# WM-QD-4A external-memory/shared-slot/QH quality contract
# ---------------------------------------------------------------------------

def wm_qd4a_external_memory_contract() -> dict:
    """Return serialization-safe quality metadata for this external-memory layer.

    This no-mutation contract declares external memory response schemas, MANN
    trace visibility, fusion shape checks, shared-slot ownership/conflict
    metadata, QH code schema validation, interference checks, trace
    serialization, PAAMA-X write-permission metadata, fallback behavior, and
    compatibility with QDTWorkingMemory.
    """
    return external_memory_contract_trace(module=__name__)
