from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence
import hashlib

import torch

from .wm_shared_slot_registry import SharedSlotRegistry, SharedSlotRecord, canonical_slot_id


def tensor_fingerprint(x: torch.Tensor, max_values: int = 64) -> str:
    """Stable-ish fingerprint for metadata linkage, not cryptographic content storage."""
    if x.numel() == 0:
        return "empty"
    flat = x.detach().float().cpu().reshape(-1)[:max_values]
    payload = ",".join(f"{float(v):.6f}" for v in flat).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


@dataclass
class MirroredContentRule:
    source_memory: str
    target_memory: str
    mirror_mode: str = "metadata_only"
    allow_content_copy: bool = False
    require_conflict_check: bool = True
    reason: str = "shared_slot_doctrine"

    def validate(self) -> None:
        if self.source_memory not in {"ltm", "mann", "wm", "spcp"}:
            raise ValueError("invalid source_memory")
        if self.target_memory not in {"ltm", "mann", "wm", "spcp"}:
            raise ValueError("invalid target_memory")
        if self.mirror_mode not in {"metadata_only", "content_reference", "content_copy"}:
            raise ValueError("invalid mirror_mode")
        if self.mirror_mode == "content_copy" and not self.allow_content_copy:
            raise ValueError("content_copy mode requires allow_content_copy=True")


    def attach_qh_record(
        self,
        canonical_id: str,
        qh_record_id: str,
        composite_code: str,
        interference_detected: bool = False,
    ) -> None:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        ref = {
            "qh_record_id": qh_record_id,
            "composite_code": composite_code,
            "interference_detected": bool(interference_detected),
        }
        refs = self.qh_record_refs.setdefault(canonical_id, [])
        if ref not in refs:
            refs.append(ref)
        if interference_detected:
            self.registry.mark_conflict(canonical_id, reason="qh_interference_detected", quarantine=True)

    def qh_trace_for_slot(self, canonical_id: str) -> Dict[str, Any]:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        refs = self.qh_record_refs.get(canonical_id, [])
        return {
            "trace_type": "shared_slot_qh_reference",
            "canonical_id": canonical_id,
            "qh_refs": refs,
            "qh_record_count": len(refs),
            "paamax_metadata": {
                "trace_type": "shared_slot_qh_reference",
                "interference_detected": any(r.get("interference_detected", False) for r in refs),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SharedSlotStoreConfig:
    namespace: str = "qdt_wm"
    dim: int = 32
    default_owner: str = "shared"
    require_write_permission: bool = True

    def validate(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if self.dim <= 0:
            raise ValueError("dim must be positive")


@dataclass
class SharedSlotWriteResult:
    canonical_id: str
    memory_type: str
    local_slot_id: str
    fingerprint: str
    write_permission_required: bool
    write_permission_granted: bool
    conflict_state: str
    record: Dict[str, Any]
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)


    def attach_qh_record(
        self,
        canonical_id: str,
        qh_record_id: str,
        composite_code: str,
        interference_detected: bool = False,
    ) -> None:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        ref = {
            "qh_record_id": qh_record_id,
            "composite_code": composite_code,
            "interference_detected": bool(interference_detected),
        }
        refs = self.qh_record_refs.setdefault(canonical_id, [])
        if ref not in refs:
            refs.append(ref)
        if interference_detected:
            self.registry.mark_conflict(canonical_id, reason="qh_interference_detected", quarantine=True)

    def qh_trace_for_slot(self, canonical_id: str) -> Dict[str, Any]:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        refs = self.qh_record_refs.get(canonical_id, [])
        return {
            "trace_type": "shared_slot_qh_reference",
            "canonical_id": canonical_id,
            "qh_refs": refs,
            "qh_record_count": len(refs),
            "paamax_metadata": {
                "trace_type": "shared_slot_qh_reference",
                "interference_detected": any(r.get("interference_detected", False) for r in refs),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SharedSlotStore:
    """Shared canonical slot store for LTM/MANN mirrored content doctrine.

    WM-4B purpose:
    - assign canonical shared slot IDs.
    - track mirrored LTM/MANN references without merging their actual memory
      subsystems.
    - carry ownership/source/conflict/write-permission metadata.
    """

    def __init__(self, config: SharedSlotStoreConfig):
        config.validate()
        self.config = config
        self.registry = SharedSlotRegistry(namespace=config.namespace)
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.qh_record_refs: Dict[str, List[Dict[str, Any]]] = {}
        self.rules: List[MirroredContentRule] = [
            MirroredContentRule("ltm", "mann", mirror_mode="metadata_only", reason="canonical_fact_visible_to_reasoning"),
            MirroredContentRule("mann", "ltm", mirror_mode="metadata_only", reason="reasoning_trace_visible_to_stable_memory"),
            MirroredContentRule("spcp", "mann", mirror_mode="metadata_only", reason="procedure_visible_to_reasoning"),
        ]

    def validate_rules(self) -> None:
        for rule in self.rules:
            rule.validate()

    def write_slot(
        self,
        *,
        memory_type: str,
        local_slot_id: str,
        content: torch.Tensor,
        owner: Optional[str] = None,
        geometry_map: Optional[str] = None,
        depth_index: Optional[int] = None,
        confidence: float = 1.0,
        write_permission: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedSlotWriteResult:
        if content.dim() != 1 or content.size(0) != self.config.dim:
            raise ValueError(f"content must be [D={self.config.dim}]")
        if not torch.isfinite(content).all():
            raise ValueError("content contains NaN or Inf")
        fp = tensor_fingerprint(content)
        record = self.registry.get_or_create(
            local_slot_id=local_slot_id,
            memory_type=memory_type,
            owner=owner or self.config.default_owner,
            content_fingerprint=fp,
            geometry_map=geometry_map,
            depth_index=depth_index,
            confidence=confidence,
            metadata=metadata or {},
        )
        if self.config.require_write_permission:
            record.grant_write_permission(write_permission)
        else:
            record.grant_write_permission(True)
        self.embeddings[record.canonical_id] = content.detach().clone()
        record.validate()
        return SharedSlotWriteResult(
            canonical_id=record.canonical_id,
            memory_type=memory_type,
            local_slot_id=local_slot_id,
            fingerprint=fp,
            write_permission_required=self.config.require_write_permission,
            write_permission_granted=record.write_permission_granted,
            conflict_state=record.conflict_state,
            record=record.to_dict(),
            paamax_metadata={
                "trace_type": "shared_slot_write",
                "canonical_id": record.canonical_id,
                "write_permission_required": self.config.require_write_permission,
                "write_permission_granted": record.write_permission_granted,
                "conflict_state": record.conflict_state,
                "owner": record.owner,
            },
        )

    def mirror_slot(
        self,
        canonical_id: str,
        *,
        memory_type: str,
        local_slot_id: str,
        geometry_map: Optional[str] = None,
        depth_index: Optional[int] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedSlotRecord:
        record = self.registry.link_mirror(
            canonical_id,
            memory_type=memory_type,
            local_slot_id=local_slot_id,
            geometry_map=geometry_map,
            depth_index=depth_index,
            confidence=confidence,
            metadata=metadata or {},
        )
        return record

    def mark_conflict(self, canonical_id: str, reason: str, quarantine: bool = False) -> SharedSlotRecord:
        return self.registry.mark_conflict(canonical_id, reason=reason, quarantine=quarantine)

    def get_content(self, canonical_id: str) -> Optional[torch.Tensor]:
        content = self.embeddings.get(canonical_id)
        return None if content is None else content.detach().clone()

    def references_for_memory(self, memory_type: str) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.registry.by_memory_type(memory_type)]

    def trace_for_local_slot(self, memory_type: str, local_slot_id: str) -> Dict[str, Any]:
        records = []
        for record in self.registry.records.values():
            for mirror in record.mirrors:
                if mirror.memory_type == memory_type and mirror.local_slot_id == local_slot_id:
                    records.append(record.to_dict())
                    break
        return {
            "trace_type": "shared_slot_reference",
            "memory_type": memory_type,
            "local_slot_id": local_slot_id,
            "records": records,
            "canonical_ids": [r["canonical_id"] for r in records],
            "paamax_metadata": {
                "trace_type": "shared_slot_reference",
                "write_permission_required": self.config.require_write_permission,
                "conflict_states": [r["conflict_state"] for r in records],
            },
        }


    def attach_qh_record(
        self,
        canonical_id: str,
        qh_record_id: str,
        composite_code: str,
        interference_detected: bool = False,
    ) -> None:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        ref = {
            "qh_record_id": qh_record_id,
            "composite_code": composite_code,
            "interference_detected": bool(interference_detected),
        }
        refs = self.qh_record_refs.setdefault(canonical_id, [])
        if ref not in refs:
            refs.append(ref)
        if interference_detected:
            self.registry.mark_conflict(canonical_id, reason="qh_interference_detected", quarantine=True)

    def qh_trace_for_slot(self, canonical_id: str) -> Dict[str, Any]:
        if canonical_id not in self.registry.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        refs = self.qh_record_refs.get(canonical_id, [])
        return {
            "trace_type": "shared_slot_qh_reference",
            "canonical_id": canonical_id,
            "qh_refs": refs,
            "qh_record_count": len(refs),
            "paamax_metadata": {
                "trace_type": "shared_slot_qh_reference",
                "interference_detected": any(r.get("interference_detected", False) for r in refs),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "rules": [r.to_dict() for r in self.rules],
            "registry": self.registry.to_dict(),
            "embedding_ids": sorted(self.embeddings.keys()),
            "qh_record_refs": self.qh_record_refs,
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
