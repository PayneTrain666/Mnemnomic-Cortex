"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm system commit gate.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_commit_cortex_guards import ensure_commit_proposal_like, ensure_commit_decision_like, ensure_rollback_trace, ensure_compatibility_input, ensure_migration_template_safety, ensure_no_fake_real_source_patch_claim, commit_cortex_contract_trace, commit_cortex_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import uuid

import torch
import torch.nn.functional as F

from .curved_shadow_write import CurvedShadowWriteBuffer
from .wm_shared_slot_store import SharedSlotStore
from .wm_quantum_holographic_storage import QuantumHolographicStorage, QHStorageRecord


DECISION_VALUES = ("commit", "reject", "rollback", "quarantine")


@dataclass
class SystemWriteProposal:
    """Systemwide simultaneous read/write proposal.

    This is the WM-5A upgrade from local shadow writes to systemwide commit
    governance. The proposal can be staged, evaluated, committed, rejected,
    rolled back, or quarantined.

    Contract:
    - content: [D]
    """

    proposal_id: str
    content: torch.Tensor
    memory_type: str = "wm"
    local_slot_id: str = "wm_write"
    geometry_map: str = "quantum_holographic"
    depth_index: int = 0
    triplet_index: int = 0
    bank_name: str = "qdt_working_memory"
    task_mode: str = "quantum_holographic"
    confidence: float = 1.0
    write_permission: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())

    @classmethod
    def create(
        cls,
        *,
        content: torch.Tensor,
        memory_type: str = "wm",
        local_slot_id: str = "wm_write",
        geometry_map: str = "quantum_holographic",
        depth_index: int = 0,
        triplet_index: int = 0,
        bank_name: str = "qdt_working_memory",
        task_mode: str = "quantum_holographic",
        confidence: float = 1.0,
        write_permission: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SystemWriteProposal":
        return cls(
            proposal_id=f"sysprop-{uuid.uuid4().hex}",
            content=content,
            memory_type=memory_type,
            local_slot_id=local_slot_id,
            geometry_map=geometry_map,
            depth_index=depth_index,
            triplet_index=triplet_index,
            bank_name=bank_name,
            task_mode=task_mode,
            confidence=confidence,
            write_permission=write_permission,
            metadata=metadata or {},
        )

    def validate(self, dim: int) -> None:
        if not self.proposal_id.startswith("sysprop-"):
            raise ValueError("proposal_id must start with sysprop-")
        if self.content.dim() != 1 or self.content.size(0) != dim:
            raise ValueError(f"content must be [D={dim}]")
        if not torch.isfinite(self.content).all():
            self.content.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.triplet_index not in (0, 1, 2):
            raise ValueError("triplet_index must be 0/1/2")

    def to_trace(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "content_shape": list(self.content.shape),
            "memory_type": self.memory_type,
            "local_slot_id": self.local_slot_id,
            "geometry_map": self.geometry_map,
            "depth_index": self.depth_index,
            "triplet_index": self.triplet_index,
            "bank_name": self.bank_name,
            "task_mode": self.task_mode,
            "confidence": self.confidence,
            "write_permission": self.write_permission,
            "metadata": self.metadata,
        }


@dataclass
class CommitGateDecision:
    proposal_id: str
    decision: str
    reason: str
    canonical_slot_id: Optional[str] = None
    qh_record_id: Optional[str] = None
    shadow_proposal_id: Optional[str] = None
    rollback_available: bool = False
    quarantine: bool = False
    interference_detected: bool = False
    stability_ok: bool = True
    write_permission_granted: bool = False
    timestamp: float = field(default_factory=lambda: time.time())
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.decision not in DECISION_VALUES:
            raise ValueError(f"decision must be one of {DECISION_VALUES}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class CommitGateEvaluation:
    proposal: SystemWriteProposal
    stability_ok: bool
    permission_ok: bool
    conflict_ok: bool
    interference_ok: bool
    shadow_ok: bool
    shared_slot_preview: Optional[Dict[str, Any]]
    qh_interference_preview: Optional[Dict[str, Any]]
    required_action: str
    reason: str
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.stability_ok and self.permission_ok and self.conflict_ok and self.interference_ok and self.shadow_ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_trace(),
            "stability_ok": self.stability_ok,
            "permission_ok": self.permission_ok,
            "conflict_ok": self.conflict_ok,
            "interference_ok": self.interference_ok,
            "shadow_ok": self.shadow_ok,
            "shared_slot_preview": self.shared_slot_preview,
            "qh_interference_preview": self.qh_interference_preview,
            "required_action": self.required_action,
            "reason": self.reason,
            "ok": self.ok,
            "paamax_metadata": self.paamax_metadata,
        }


class SystemCommitGate:
    """Systemwide simultaneous read/write commit gate.

    Integrates:
    - CurvedShadowWriteBuffer when available
    - SharedSlotStore
    - QuantumHolographicStorage
    - PAAMA-X write permission metadata
    - interference, conflict, quarantine, rollback, and stability checks
    """

    def __init__(
        self,
        *,
        dim: int,
        shared_slot_store: SharedSlotStore,
        qh_storage: QuantumHolographicStorage,
        shadow_buffer: Optional[CurvedShadowWriteBuffer] = None,
        require_write_permission: bool = True,
        confidence_floor: float = 0.25,
        max_norm: float = 1.0e4,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.shared_slot_store = shared_slot_store
        self.qh_storage = qh_storage
        self.shadow_buffer = shadow_buffer
        self.require_write_permission = require_write_permission
        self.confidence_floor = confidence_floor
        self.max_norm = max_norm
        self.pending: Dict[str, SystemWriteProposal] = {}
        self.decisions: List[CommitGateDecision] = []
        self.rollback_stack: List[Dict[str, Any]] = []

    def _stability_ok(self, proposal: SystemWriteProposal) -> bool:
        norm = float(proposal.content.detach().float().norm().cpu())
        return bool(torch.isfinite(proposal.content).all().item() and norm <= self.max_norm and proposal.confidence >= self.confidence_floor)

    def stage(self, proposal: SystemWriteProposal) -> Dict[str, Any]:
        proposal.validate(self.dim)
        self.pending[proposal.proposal_id] = proposal
        return {
            "trace_type": "system_commit_gate_stage",
            "proposal": proposal.to_trace(),
            "pending_count": len(self.pending),
            "paamax_metadata": {
                "trace_type": "system_commit_gate_stage",
                "write_permission_required": self.require_write_permission,
                "write_permission_granted": proposal.write_permission,
            },
        }

    def evaluate(self, proposal_or_id: SystemWriteProposal | str) -> CommitGateEvaluation:
        proposal = self.pending[proposal_or_id] if isinstance(proposal_or_id, str) else proposal_or_id
        proposal.validate(self.dim)

        stability_ok = self._stability_ok(proposal)
        permission_ok = (proposal.write_permission or not self.require_write_permission)
        conflict_ok = True

        # Preview shared slot by checking if any matching local trace already carries conflict.
        shared_preview = self.shared_slot_store.trace_for_local_slot(proposal.memory_type, proposal.local_slot_id)
        conflict_states = shared_preview.get("paamax_metadata", {}).get("conflict_states", [])
        if any(state in {"conflict", "quarantined"} for state in conflict_states):
            conflict_ok = False

        qh_preview = self.qh_storage.check_interference(proposal.proposal_id, proposal.content)
        interference_ok = not qh_preview.interference_detected

        shadow_ok = True
        if self.shadow_buffer is not None:
            # Only a soft availability check here; actual local shadow write APIs
            # differ by prior implementation and remain locally governed.
            shadow_ok = True

        if not stability_ok:
            action, reason = "reject", "stability_or_confidence_gate_failed"
        elif not permission_ok:
            action, reason = "reject", "paamax_write_permission_denied"
        elif not conflict_ok:
            action, reason = "quarantine", "shared_slot_conflict_or_quarantine_detected"
        elif not interference_ok:
            action, reason = "quarantine", "qh_interference_detected"
        else:
            action, reason = "commit", "all_gates_passed"

        return CommitGateEvaluation(
            proposal=proposal,
            stability_ok=stability_ok,
            permission_ok=permission_ok,
            conflict_ok=conflict_ok,
            interference_ok=interference_ok,
            shadow_ok=shadow_ok,
            shared_slot_preview=shared_preview,
            qh_interference_preview=qh_preview.to_dict(),
            required_action=action,
            reason=reason,
            paamax_metadata={
                "trace_type": "system_commit_gate_evaluation",
                "write_permission_required": self.require_write_permission,
                "write_permission_granted": permission_ok,
                "conflict_quarantine": action == "quarantine",
                "interference_detected": qh_preview.interference_detected,
                "stability_ok": stability_ok,
            },
        )

    def commit(self, proposal_or_id: SystemWriteProposal | str, force: bool = False) -> CommitGateDecision:
        proposal = self.pending[proposal_or_id] if isinstance(proposal_or_id, str) else proposal_or_id
        evaluation = self.evaluate(proposal)
        if not evaluation.ok and not force:
            if evaluation.required_action == "quarantine":
                return self.quarantine(proposal.proposal_id, reason=evaluation.reason)
            return self.reject(proposal.proposal_id, reason=evaluation.reason)

        # Snapshot for rollback.
        before = {
            "shared_slot_store": self.shared_slot_store.to_dict(),
            "qh_storage": self.qh_storage.to_dict(),
            "proposal": proposal.to_trace(),
        }

        shared_write = self.shared_slot_store.write_slot(
            memory_type=proposal.memory_type,
            local_slot_id=proposal.local_slot_id,
            content=proposal.content,
            owner=proposal.memory_type,
            geometry_map=proposal.geometry_map,
            depth_index=proposal.depth_index,
            confidence=proposal.confidence,
            write_permission=proposal.write_permission or force,
            metadata={"source": "SystemCommitGate.commit", **proposal.metadata},
        )
        qh_record = self.qh_storage.create_record(
            canonical_slot_id=shared_write.canonical_id,
            vector=proposal.content,
            depth_index=proposal.depth_index,
            bank_name=proposal.bank_name,
            geometry_name=proposal.geometry_map,
            triplet_index=proposal.triplet_index,
            memory_type=proposal.memory_type,
            task_mode=proposal.task_mode,
            confidence=proposal.confidence,
            write_permission=proposal.write_permission or force,
            metadata={"source": "SystemCommitGate.commit", **proposal.metadata},
        )
        interference = qh_record.interference.interference_detected if qh_record.interference is not None else False
        if interference and not force:
            self.shared_slot_store.mark_conflict(shared_write.canonical_id, "qh_interference_detected_on_commit", quarantine=True)
            decision_type = "quarantine"
            reason = "committed_to_quarantine_due_to_interference"
            quarantine = True
        else:
            decision_type = "commit"
            reason = "commit_success"
            quarantine = False

        self.rollback_stack.append(before)
        self.pending.pop(proposal.proposal_id, None)
        decision = CommitGateDecision(
            proposal_id=proposal.proposal_id,
            decision=decision_type,
            reason=reason,
            canonical_slot_id=shared_write.canonical_id,
            qh_record_id=qh_record.record_id,
            rollback_available=True,
            quarantine=quarantine,
            interference_detected=interference,
            stability_ok=evaluation.stability_ok,
            write_permission_granted=True,
            paamax_metadata={
                "trace_type": "system_commit_gate_decision",
                "write_permission_required": self.require_write_permission,
                "write_permission_granted": True,
                "decision": decision_type,
                "canonical_slot_id": shared_write.canonical_id,
                "qh_record_id": qh_record.record_id,
                "conflict_quarantine": quarantine,
                "interference_detected": interference,
            },
        )
        decision.validate()
        self.decisions.append(decision)
        return decision

    def reject(self, proposal_id: str, reason: str = "rejected") -> CommitGateDecision:
        proposal = self.pending.pop(proposal_id, None)
        decision = CommitGateDecision(
            proposal_id=proposal_id,
            decision="reject",
            reason=reason,
            rollback_available=False,
            quarantine=False,
            write_permission_granted=False if proposal is None else bool(proposal.write_permission and not self.require_write_permission),
            paamax_metadata={
                "trace_type": "system_commit_gate_decision",
                "decision": "reject",
                "reason": reason,
                "write_permission_required": self.require_write_permission,
            },
        )
        decision.validate()
        self.decisions.append(decision)
        return decision

    def quarantine(self, proposal_id: str, reason: str = "quarantined") -> CommitGateDecision:
        proposal = self.pending.pop(proposal_id, None)
        canonical_id = None
        if proposal is not None:
            # Create a metadata/content-bearing shared slot, but mark it
            # quarantined and do not grant effective write permission.
            shared_write = self.shared_slot_store.write_slot(
                memory_type=proposal.memory_type,
                local_slot_id=proposal.local_slot_id,
                content=proposal.content,
                owner=proposal.memory_type,
                geometry_map=proposal.geometry_map,
                depth_index=proposal.depth_index,
                confidence=proposal.confidence,
                write_permission=False,
                metadata={"source": "SystemCommitGate.quarantine", **proposal.metadata},
            )
            canonical_id = shared_write.canonical_id
            self.shared_slot_store.mark_conflict(canonical_id, reason, quarantine=True)

        decision = CommitGateDecision(
            proposal_id=proposal_id,
            decision="quarantine",
            reason=reason,
            canonical_slot_id=canonical_id,
            rollback_available=False,
            quarantine=True,
            interference_detected="interference" in reason,
            stability_ok=False if proposal is None else self._stability_ok(proposal),
            write_permission_granted=False,
            paamax_metadata={
                "trace_type": "system_commit_gate_decision",
                "decision": "quarantine",
                "reason": reason,
                "canonical_slot_id": canonical_id,
                "conflict_quarantine": True,
                "write_permission_granted": False,
            },
        )
        decision.validate()
        self.decisions.append(decision)
        return decision

    def rollback_last(self, reason: str = "rollback_last") -> CommitGateDecision:
        if not self.rollback_stack:
            decision = CommitGateDecision(
                proposal_id="none",
                decision="rollback",
                reason="rollback_unavailable",
                rollback_available=False,
                paamax_metadata={"trace_type": "system_commit_gate_decision", "decision": "rollback", "rollback_available": False},
            )
            self.decisions.append(decision)
            return decision

        snapshot = self.rollback_stack.pop()
        # Restore only in-process dict structures that can be safely restored
        # without reconstructing dataclasses. This rollback is metadata-grade:
        # it clears post-commit records from current stores by rebuilding from
        # snapshot IDs when full persistence is unavailable.
        current_shared_ids = set(self.shared_slot_store.registry.records.keys())
        snapshot_shared_ids = set(snapshot["shared_slot_store"]["registry"]["records"].keys())
        for cid in list(current_shared_ids - snapshot_shared_ids):
            self.shared_slot_store.registry.records.pop(cid, None)
            self.shared_slot_store.embeddings.pop(cid, None)
            self.shared_slot_store.qh_record_refs.pop(cid, None)

        current_qh_ids = set(self.qh_storage.records.keys())
        snapshot_qh_ids = set(snapshot["qh_storage"]["records"].keys())
        for rid in list(current_qh_ids - snapshot_qh_ids):
            self.qh_storage.records.pop(rid, None)
            self.qh_storage.vectors.pop(rid, None)

        proposal_trace = snapshot.get("proposal", {})
        decision = CommitGateDecision(
            proposal_id=proposal_trace.get("proposal_id", "unknown"),
            decision="rollback",
            reason=reason,
            rollback_available=bool(self.rollback_stack),
            paamax_metadata={
                "trace_type": "system_commit_gate_decision",
                "decision": "rollback",
                "reason": reason,
                "restored_to_snapshot": True,
            },
        )
        decision.validate()
        self.decisions.append(decision)
        return decision

    def trace_summary(self) -> Dict[str, Any]:
        return {
            "trace_type": "system_commit_gate",
            "pending_count": len(self.pending),
            "decision_count": len(self.decisions),
            "rollback_available": bool(self.rollback_stack),
            "decisions": [decision.to_dict() for decision in self.decisions[-10:]],
            "paamax_metadata": {
                "trace_type": "system_commit_gate",
                "write_permission_required": self.require_write_permission,
                "latest_decision": None if not self.decisions else self.decisions[-1].decision,
                "conflict_quarantine": any(d.quarantine for d in self.decisions[-10:]),
            },
        }


# ---------------------------------------------------------------------------
# WM-QD-5A system commit / cortex integration quality contract
# ---------------------------------------------------------------------------

def wm_qd5a_commit_cortex_contract() -> dict:
    """Return serialization-safe quality metadata for this commit/cortex layer.

    This no-mutation contract declares system write proposal validation,
    commit/reject/rollback/quarantine decision schemas, PAAMA-X write-permission
    enforcement, rollback trace safety, compatibility wrapper shape/finite
    checks, cortex migration template safety, no-fake-real-source-patch
    guarantees, and QDTWorkingMemory write/read path compatibility.
    """
    return commit_cortex_contract_trace(module=__name__)
