from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import time

from .depth_lattice_types import _safe_jsonable


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SharedDepthSlotRegistryError(ValueError):
    """Raised when shared-depth slot registry input is invalid."""


@dataclass
class SharedDepthSlotRecord:
    """Canonical identity record for mirrored WM/MANN/LTM depth slots.

    This record intentionally stores references and metadata only. It never
    couples physical tensors across WM, MANN, or LTM.
    """

    canonical_slot_id: str
    content_hash: str
    wm_refs: List[str] = field(default_factory=list)
    mann_refs: List[str] = field(default_factory=list)
    ltm_refs: List[str] = field(default_factory=list)
    project_id: Optional[str] = None
    chat_id: Optional[str] = None
    episode_id: Optional[str] = None
    depth_roles_present: List[int] = field(default_factory=list)
    confidence: float = 1.0
    disagreement: float = 0.0
    source_stage: Optional[str] = None
    source_pack: Optional[str] = None
    consolidation_status: str = "unconsolidated"
    registry_lineage: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return _safe_jsonable(
            {
                "canonical_slot_id": self.canonical_slot_id,
                "content_hash": self.content_hash,
                "wm_refs": self.wm_refs,
                "mann_refs": self.mann_refs,
                "ltm_refs": self.ltm_refs,
                "project_id": self.project_id,
                "chat_id": self.chat_id,
                "episode_id": self.episode_id,
                "depth_roles_present": sorted(set(int(x) for x in self.depth_roles_present)),
                "confidence": self.confidence,
                "disagreement": self.disagreement,
                "source_stage": self.source_stage,
                "source_pack": self.source_pack,
                "consolidation_status": self.consolidation_status,
                "registry_lineage": self.registry_lineage,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "safety": {
                    "shared_physical_tensor": False,
                    "canonical_identity_only": True,
                    "no_direct_ltm_mann_tensor_coupling": True,
                    "no_direct_wm_ltm_tensor_coupling": True,
                    "no_direct_wm_mann_tensor_coupling": True,
                },
            }
        )


@dataclass
class SharedDepthSlotRegistry:
    records: Dict[str, SharedDepthSlotRecord] = field(default_factory=dict)

    def create_or_update(
        self,
        *,
        canonical_slot_id: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
        wm_ref: Optional[str] = None,
        mann_ref: Optional[str] = None,
        ltm_ref: Optional[str] = None,
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        depth_roles_present: Optional[List[int]] = None,
        confidence: float = 1.0,
        disagreement: float = 0.0,
        source_stage: Optional[str] = None,
        source_pack: Optional[str] = None,
        consolidation_status: Optional[str] = None,
        registry_lineage: Optional[List[Dict[str, Any]]] = None,
    ) -> SharedDepthSlotRecord:
        if not canonical_slot_id:
            raise SharedDepthSlotRegistryError("canonical_slot_id is required")
        if content_hash is None:
            if content is None:
                raise SharedDepthSlotRegistryError("content or content_hash is required")
            content_hash = _hash_content(content)
        if not (0.0 <= confidence <= 1.0):
            raise SharedDepthSlotRegistryError("confidence must be in [0,1]")
        if not (0.0 <= disagreement <= 1.0):
            raise SharedDepthSlotRegistryError("disagreement must be in [0,1]")
        if depth_roles_present:
            for d in depth_roles_present:
                if not (0 <= int(d) < 8):
                    raise SharedDepthSlotRegistryError("depth roles must be in [0,7]")
        if consolidation_status is not None and consolidation_status not in {
            "unconsolidated",
            "shadow_proposed",
            "ready_for_review",
            "committed",
            "rejected",
            "quarantined",
        }:
            raise SharedDepthSlotRegistryError("unsupported consolidation_status")

        rec = self.records.get(canonical_slot_id)
        if rec is None:
            rec = SharedDepthSlotRecord(
                canonical_slot_id=canonical_slot_id,
                content_hash=content_hash,
                project_id=project_id,
                chat_id=chat_id,
                episode_id=episode_id,
                depth_roles_present=list(depth_roles_present or []),
                confidence=confidence,
                disagreement=disagreement,
                source_stage=source_stage,
                source_pack=source_pack,
                consolidation_status=consolidation_status or "unconsolidated",
                registry_lineage=list(registry_lineage or []),
            )
            self.records[canonical_slot_id] = rec
        else:
            if rec.content_hash != content_hash:
                rec.disagreement = max(rec.disagreement, disagreement, 0.5)
            rec.project_id = project_id or rec.project_id
            rec.chat_id = chat_id or rec.chat_id
            rec.episode_id = episode_id or rec.episode_id
            rec.confidence = max(0.0, min(1.0, confidence))
            rec.disagreement = max(0.0, min(1.0, max(rec.disagreement, disagreement)))
            rec.depth_roles_present = sorted(set(rec.depth_roles_present + list(depth_roles_present or [])))
            rec.source_stage = source_stage or rec.source_stage
            rec.source_pack = source_pack or rec.source_pack
            if consolidation_status is not None:
                rec.consolidation_status = consolidation_status
            if registry_lineage:
                rec.registry_lineage.extend(registry_lineage)
            rec.updated_at = time.time()

        if wm_ref and wm_ref not in rec.wm_refs:
            rec.wm_refs.append(wm_ref)
        if mann_ref and mann_ref not in rec.mann_refs:
            rec.mann_refs.append(mann_ref)
        if ltm_ref and ltm_ref not in rec.ltm_refs:
            rec.ltm_refs.append(ltm_ref)
        return rec

    def propose_consolidation(
        self,
        *,
        canonical_slot_id: str,
        target_ltm_ref: str,
        source_stage: str,
        source_pack: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rec = self.records.get(canonical_slot_id)
        if rec is None:
            raise SharedDepthSlotRegistryError("canonical_slot_id not found")
        if target_ltm_ref not in rec.ltm_refs:
            rec.ltm_refs.append(target_ltm_ref)
        rec.consolidation_status = "shadow_proposed"
        lineage_item = {
            "event": "shadow_consolidation_proposed",
            "source_stage": source_stage,
            "source_pack": source_pack,
            "target_ltm_ref": target_ltm_ref,
            "evidence": evidence or {},
            "created_at": time.time(),
        }
        rec.registry_lineage.append(lineage_item)
        rec.updated_at = time.time()
        return {
            "committed": False,
            "shadow_only": True,
            "canonical_slot_id": canonical_slot_id,
            "target_ltm_ref": target_ltm_ref,
            "consolidation_status": rec.consolidation_status,
            "lineage_item": _safe_jsonable(lineage_item),
            "safety": {
                "shared_physical_tensor": False,
                "permanent_consolidation_requires_gate": True,
            },
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": True,
                "write_permission_granted": False,
                "no_memory_store_mutation": True,
            },
        }

    def get(self, canonical_slot_id: str) -> Optional[SharedDepthSlotRecord]:
        return self.records.get(canonical_slot_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_count": len(self.records),
            "records": {k: v.to_dict() for k, v in sorted(self.records.items())},
            "safety": {
                "shared_physical_tensor": False,
                "canonical_identity_only": True,
                "no_direct_ltm_mann_tensor_coupling": True,
            },
        }


def shared_depth_registry_contract() -> Dict[str, Any]:
    return {
        "module": "shared_depth_slot_registry",
        "stage": "REASON-1D",
        "canonical_slot_ids": True,
        "wm_refs": True,
        "mann_refs": True,
        "ltm_refs": True,
        "provenance_fields": [
            "source_stage",
            "source_pack",
            "consolidation_status",
            "registry_lineage",
        ],
        "shared_physical_tensor": False,
        "shadow_consolidation_only_by_default": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
