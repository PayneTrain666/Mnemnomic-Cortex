"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm shared slot registry.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import time


ALLOWED_OWNERS = ("wm", "ltm", "mann", "spcp", "shared")
ALLOWED_MEMORY_TYPES = ("wm", "ltm", "mann", "spcp")


def canonical_slot_id(namespace: str, local_slot_id: str, content_fingerprint: Optional[str] = None) -> str:
    """Create a deterministic canonical shared-slot ID.

    The ID is stable for the same namespace/local slot/fingerprint tuple and
    intentionally does not expose raw content.
    """
    if not namespace:
        raise ValueError("namespace must be non-empty")
    if not local_slot_id:
        raise ValueError("local_slot_id must be non-empty")
    key = f"{namespace}|{local_slot_id}|{content_fingerprint or ''}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:24]
    return f"css-{digest}"


@dataclass
class SharedSlotMirrorRef:
    """Reference to the same canonical content as viewed through a memory system."""

    memory_type: str
    local_slot_id: str
    geometry_map: Optional[str] = None
    depth_index: Optional[int] = None
    confidence: float = 1.0
    last_seen: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.memory_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {ALLOWED_MEMORY_TYPES}")
        if not self.local_slot_id:
            raise ValueError("local_slot_id must be non-empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SharedSlotRecord:
    """Canonical shared-slot record.

    This is the metadata layer that lets LTM and MANN share slots without
    collapsing their separate subsystem identities.
    """

    canonical_id: str
    namespace: str
    owner: str = "shared"
    content_fingerprint: Optional[str] = None
    mirrors: List[SharedSlotMirrorRef] = field(default_factory=list)
    source_memory_types: List[str] = field(default_factory=list)
    conflict_state: str = "clear"
    conflict_reason: Optional[str] = None
    write_permission_required: bool = True
    write_permission_granted: bool = False
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.canonical_id.startswith("css-"):
            raise ValueError("canonical_id must start with css-")
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if self.owner not in ALLOWED_OWNERS:
            raise ValueError(f"owner must be one of {ALLOWED_OWNERS}")
        for mirror in self.mirrors:
            mirror.validate()
        if self.conflict_state not in {"clear", "suspected", "conflict", "quarantined"}:
            raise ValueError("invalid conflict_state")
        for mt in self.source_memory_types:
            if mt not in ALLOWED_MEMORY_TYPES:
                raise ValueError(f"invalid source memory type: {mt}")

    def add_or_update_mirror(self, mirror: SharedSlotMirrorRef) -> None:
        mirror.validate()
        for i, existing in enumerate(self.mirrors):
            if existing.memory_type == mirror.memory_type and existing.local_slot_id == mirror.local_slot_id:
                self.mirrors[i] = mirror
                break
        else:
            self.mirrors.append(mirror)
        if mirror.memory_type not in self.source_memory_types:
            self.source_memory_types.append(mirror.memory_type)
        self.updated_at = time.time()

    def set_conflict(self, state: str, reason: Optional[str] = None) -> None:
        if state not in {"clear", "suspected", "conflict", "quarantined"}:
            raise ValueError("invalid conflict state")
        self.conflict_state = state
        self.conflict_reason = reason
        self.updated_at = time.time()

    def grant_write_permission(self, granted: bool = True) -> None:
        self.write_permission_granted = bool(granted)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["mirrors"] = [m.to_dict() for m in self.mirrors]
        return out


class SharedSlotRegistry:
    """Registry of canonical shared slot records.

    The registry is intentionally pure-Python/serializable at WM-4B. A future
    persistence adapter can map this to database/storage without changing the
    contract.
    """

    def __init__(self, namespace: str = "qdt_wm"):
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self.namespace = namespace
        self.records: Dict[str, SharedSlotRecord] = {}

    def get_or_create(
        self,
        local_slot_id: str,
        *,
        memory_type: str,
        owner: str = "shared",
        content_fingerprint: Optional[str] = None,
        geometry_map: Optional[str] = None,
        depth_index: Optional[int] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedSlotRecord:
        if memory_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {ALLOWED_MEMORY_TYPES}")
        cid = canonical_slot_id(self.namespace, local_slot_id, content_fingerprint)
        record = self.records.get(cid)
        if record is None:
            record = SharedSlotRecord(
                canonical_id=cid,
                namespace=self.namespace,
                owner=owner,
                content_fingerprint=content_fingerprint,
                metadata=metadata or {},
            )
            self.records[cid] = record
        record.add_or_update_mirror(
            SharedSlotMirrorRef(
                memory_type=memory_type,
                local_slot_id=local_slot_id,
                geometry_map=geometry_map,
                depth_index=depth_index,
                confidence=confidence,
                metadata=metadata or {},
            )
        )
        record.validate()
        return record

    def link_mirror(
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
        if canonical_id not in self.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        record = self.records[canonical_id]
        record.add_or_update_mirror(
            SharedSlotMirrorRef(
                memory_type=memory_type,
                local_slot_id=local_slot_id,
                geometry_map=geometry_map,
                depth_index=depth_index,
                confidence=confidence,
                metadata=metadata or {},
            )
        )
        record.validate()
        return record

    def mark_conflict(self, canonical_id: str, reason: str, quarantine: bool = False) -> SharedSlotRecord:
        if canonical_id not in self.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        record = self.records[canonical_id]
        record.set_conflict("quarantined" if quarantine else "conflict", reason=reason)
        record.validate()
        return record

    def grant_write(self, canonical_id: str, granted: bool = True) -> SharedSlotRecord:
        if canonical_id not in self.records:
            raise KeyError(f"canonical slot not found: {canonical_id}")
        record = self.records[canonical_id]
        record.grant_write_permission(granted)
        record.validate()
        return record

    def get(self, canonical_id: str) -> Optional[SharedSlotRecord]:
        return self.records.get(canonical_id)

    def by_memory_type(self, memory_type: str) -> List[SharedSlotRecord]:
        if memory_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {ALLOWED_MEMORY_TYPES}")
        return [record for record in self.records.values() if memory_type in record.source_memory_types]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "record_count": len(self.records),
            "records": {cid: record.to_dict() for cid, record in self.records.items()},
        }

    def trace_summary(self) -> Dict[str, Any]:
        return {
            "trace_type": "shared_slot_registry",
            "namespace": self.namespace,
            "record_count": len(self.records),
            "canonical_ids": sorted(self.records.keys()),
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
