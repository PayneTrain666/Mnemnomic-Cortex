from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .shared_slot_allocator import SharedSlotAllocator
from .shared_slot_arbitrator import SharedSlotArbitrator
from .shared_slot_schema import (
    SYSTEM_TO_CODE,
    SlotMetadata,
    SlotProvenance,
    SlotWriteRequest,
    code_to_slot_state,
    slot_state_to_code,
    validate_system_id,
)
from .shared_slot_store import SharedSlotStore


@dataclass
class MemoryWriteOutput:
    written_slot_ids: List[int]
    action: str
    confidence: float
    diagnostics: Dict[str, Any]


class MemoryWriteEngine:
    """
    Shared-slot write orchestration over allocator + arbitrator + store.

    Doctrine:
      - single vector write normalized to [1, D]
      - write path always logs provenance
      - no durable write without explicit requested state OR lifecycle approval
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        allocator: SharedSlotAllocator,
        arbitrator: SharedSlotArbitrator,
    ) -> None:
        self.store = store
        self.allocator = allocator
        self.arbitrator = arbitrator
        self._write_trace: List[Dict[str, Any]] = []

    def _log_write_event(
        self,
        *,
        action: str,
        request: SlotWriteRequest,
        input_shape: List[int],
        chosen_slot_ids: List[int],
        reason: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        self._write_trace.append(
            {
                "event_index": len(self._write_trace) + 1,
                "version_counter": int(self.store.version_counter),
                "action": action,
                "requester_system": request.requester_system,
                "requested_state": request.requested_state,
                "requested_memory_type": request.requested_memory_type,
                "input_shape": list(input_shape),
                "chosen_slot_ids": [int(x) for x in chosen_slot_ids],
                "reason": reason,
                "diagnostics": dict(diagnostics),
            }
        )

    def get_write_trace(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._write_trace]

    def clear_write_trace(self) -> None:
        self._write_trace.clear()

    def prepare_values(self, values: torch.Tensor) -> torch.Tensor:
        v = torch.as_tensor(values, device=self.store.slot_values.device, dtype=self.store.slot_values.dtype)
        if v.dim() == 1:
            v = v.unsqueeze(0)  # [D] -> [1, D]
        if v.dim() != 2 or v.size(1) != self.store.slot_dim:
            raise ValueError(f"values must be [K, D={self.store.slot_dim}] or [D={self.store.slot_dim}]")
        if not torch.isfinite(v).all():
            raise ValueError("values contains NaN or Inf")
        return v

    def _durable_write_allowed(self, request: SlotWriteRequest) -> bool:
        if request.requested_state != "durable":
            return True
        extra = request.extra or {}
        explicit_requested_state = bool(extra.get("explicit_requested_state", False))
        lifecycle_approved = bool(extra.get("lifecycle_approved", False))
        return explicit_requested_state or lifecycle_approved

    def _build_metadata(self, slot_id: int, request: SlotWriteRequest, write_step: int) -> SlotMetadata:
        now = int(write_step)
        prior = self.store.metadata.get(int(slot_id))
        prev_promotion = 0
        prev_contradictions = 0
        created_step = now
        if isinstance(prior, SlotMetadata) and prior.provenance is not None:
            created_step = int(prior.provenance.created_step)
            prev_promotion = int(prior.provenance.promotion_count)
            prev_contradictions = int(prior.provenance.contradiction_count)

        provenance = SlotProvenance(
            source_system=request.requester_system,
            created_step=created_step,
            last_update_step=now,
            promotion_count=prev_promotion,
            contradiction_count=prev_contradictions,
            checkpoint_version=int(self.store.version_counter),
            source_trace_ids=list(request.provenance_trace_ids),
        )
        if request.requested_state == "durable":
            provenance.promotion_count += 1

        return SlotMetadata(
            slot_id=int(slot_id),
            state=request.requested_state,
            confidence=float(request.confidence),
            usage_score=float(self.store.slot_usage[int(slot_id)].item()) + 1.0,
            age_steps=0,
            primary_system_id=request.requester_system,
            semantic_tags=list(request.semantic_tags),
            memory_type=request.requested_memory_type,
            provenance=provenance,
            extra=dict(request.extra or {}),
        )

    def _system_code_for_requester(self, requester_system: str) -> int:
        try:
            validate_system_id(requester_system)
            return int(SYSTEM_TO_CODE[requester_system])
        except Exception:
            return 0

    def _default_acl_masks(
        self,
        *,
        count: int,
        requester_system: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_mask = torch.zeros(
            count,
            self.store.num_systems,
            device=self.store.slot_values.device,
            dtype=torch.bool,
        )
        write_mask = torch.zeros(
            count,
            self.store.num_systems,
            device=self.store.slot_values.device,
            dtype=torch.bool,
        )
        sys_code = self._system_code_for_requester(requester_system)
        if 0 <= sys_code < self.store.num_systems:
            read_mask[:, sys_code] = True
            write_mask[:, sys_code] = True
        return read_mask, write_mask

    def _candidate_slot_ids_for_request(self, request: SlotWriteRequest, limit: int) -> List[int]:
        scored: List[tuple[float, int]] = []
        req_mem_type = request.requested_memory_type
        for sid in range(self.store.num_slots):
            state = code_to_slot_state(int(self.store.slot_state_code[sid].item()))
            if state in {"free", "deprecated"}:
                continue
            meta = self.store.metadata.get(int(sid))
            if meta is not None and isinstance(meta, SlotMetadata):
                primary = str(meta.primary_system_id or "")
                if primary and primary != request.requester_system and state == "durable":
                    # Protect durable slots owned by other systems.
                    continue
                slot_type = getattr(meta, "memory_type", None)
                if req_mem_type is not None and slot_type not in (None, req_mem_type):
                    continue
            conf = float(self.store.slot_confidence[sid].item())
            usage = float(self.store.slot_usage[sid].item())
            scored.append((conf + 0.01 * min(usage, 10.0), int(sid)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in scored[: max(1, int(limit))]]

    def commit_new_slots(
        self,
        *,
        slot_ids: List[int],
        values: torch.Tensor,
        request: SlotWriteRequest,
    ) -> None:
        if len(slot_ids) != values.size(0):
            raise ValueError("slot_ids count must match values batch")
        ids = torch.tensor(slot_ids, device=self.store.slot_values.device, dtype=torch.long)
        confidence = torch.full((len(slot_ids),), float(request.confidence), device=self.store.slot_values.device)
        state_code = torch.full(
            (len(slot_ids),),
            slot_state_to_code(request.requested_state),
            device=self.store.slot_values.device,
            dtype=torch.long,
        )
        primary = torch.full(
            (len(slot_ids),),
            self._system_code_for_requester(request.requester_system),
            device=self.store.slot_values.device,
            dtype=torch.long,
        )
        read_mask, write_mask = self._default_acl_masks(
            count=len(slot_ids),
            requester_system=request.requester_system,
        )
        usage = torch.ones(len(slot_ids), device=self.store.slot_values.device)
        age = torch.zeros(len(slot_ids), device=self.store.slot_values.device, dtype=torch.long)

        metadata = {
            int(slot_id): self._build_metadata(int(slot_id), request, write_step=self.store.version_counter + 1)
            for slot_id in slot_ids
        }
        self.store.write_slot(ids, values, confidence, usage, age, state_code, primary, read_mask, write_mask, metadata)

    def overwrite_slots(
        self,
        *,
        slot_ids: List[int],
        values: torch.Tensor,
        request: SlotWriteRequest,
    ) -> None:
        if len(slot_ids) != values.size(0):
            raise ValueError("slot_ids count must match values batch")
        ids = torch.tensor(slot_ids, device=self.store.slot_values.device, dtype=torch.long)
        self.store.set_slot_value(
            slot_ids=ids,
            values=values,
            confidence=torch.full((len(slot_ids),), float(request.confidence), device=self.store.slot_values.device),
        )
        self.store.increment_usage(ids, amount=1.0)
        self.store.set_slot_primary_system_code(
            slot_ids=ids,
            primary_system_code=torch.full(
                (len(slot_ids),),
                self._system_code_for_requester(request.requester_system),
                device=self.store.slot_values.device,
                dtype=torch.long,
            ),
        )
        read_mask, write_mask = self._default_acl_masks(
            count=len(slot_ids),
            requester_system=request.requester_system,
        )
        self.store.set_slot_allowed_read_mask(slot_ids=ids, allowed_read_mask=read_mask)
        self.store.set_slot_allowed_write_mask(slot_ids=ids, allowed_write_mask=write_mask)
        self.store.set_slot_state_code(
            slot_ids=ids,
            state_code=torch.full(
                (len(slot_ids),),
                slot_state_to_code(request.requested_state),
                device=self.store.slot_values.device,
                dtype=torch.long,
            ),
        )
        metadata = {
            int(slot_id): self._build_metadata(int(slot_id), request, write_step=self.store.version_counter + 1)
            for slot_id in slot_ids
        }
        self.store.set_slot_metadata(slot_ids=ids, metadata=metadata)

    def merge_into_slots(
        self,
        *,
        slot_ids: List[int],
        values: torch.Tensor,
        request: SlotWriteRequest,
        merge_alpha: float = 0.5,
    ) -> None:
        if len(slot_ids) != values.size(0):
            raise ValueError("slot_ids count must match values batch")
        ids = torch.tensor(slot_ids, device=self.store.slot_values.device, dtype=torch.long)
        prior = self.store.get_slot_value(ids)
        alpha = float(max(0.0, min(1.0, merge_alpha)))
        merged = (alpha * values) + ((1.0 - alpha) * prior)
        self.overwrite_slots(slot_ids=slot_ids, values=merged, request=request)

    def write(
        self,
        *,
        request: SlotWriteRequest,
        values: torch.Tensor,  # [K, D] or [D]
    ) -> MemoryWriteOutput:
        request.validate()
        values_2d = self.prepare_values(values)
        k = int(values_2d.size(0))

        if not self._durable_write_allowed(request):
            self._log_write_event(
                action="reject",
                request=request,
                input_shape=list(values_2d.shape),
                chosen_slot_ids=[],
                reason="durable_write_denied",
                diagnostics={"doctrine": "durable_requires_explicit_or_lifecycle_approval"},
            )
            raise PermissionError(
                "durable write denied: requires explicit requested state or lifecycle approval"
            )

        candidate_count = min(max(4, k), self.store.num_slots)
        candidate_slot_ids = self._candidate_slot_ids_for_request(request, candidate_count)
        decision = self.arbitrator.decide_write(request=request, candidate_slot_ids=candidate_slot_ids)
        if bool((request.extra or {}).get("force_allocate_new", False)):
            decision.action = "allocate_new"
            decision.chosen_slot_ids = []
            decision.reason = "forced_allocate_new"
            decision.diagnostics = {**dict(decision.diagnostics), "force_allocate_new": True}

        written_slot_ids: List[int]
        if decision.action == "allocate_new":
            written_slot_ids = self.allocator.allocate(
                count=k,
                requester_system=request.requester_system,
                requested_state=request.requested_state,
            )
            self.commit_new_slots(slot_ids=written_slot_ids, values=values_2d, request=request)
        elif decision.action == "overwrite":
            if not decision.chosen_slot_ids:
                raise RuntimeError("arbitrator returned overwrite without slot ids")
            target = decision.chosen_slot_ids[:1]
            if k > 1:
                # Broadcast first chosen slot through fresh allocations for remaining rows.
                extra = self.allocator.allocate(
                    count=k - 1,
                    requester_system=request.requester_system,
                    requested_state=request.requested_state,
                )
                target = target + extra
            written_slot_ids = target
            self.overwrite_slots(slot_ids=written_slot_ids, values=values_2d, request=request)
        elif decision.action == "merge":
            if not decision.chosen_slot_ids:
                raise RuntimeError("arbitrator returned merge without slot ids")
            target = decision.chosen_slot_ids[:1]
            if k > 1:
                extra = self.allocator.allocate(
                    count=k - 1,
                    requester_system=request.requester_system,
                    requested_state=request.requested_state,
                )
                target = target + extra
            written_slot_ids = target
            self.merge_into_slots(slot_ids=written_slot_ids, values=values_2d, request=request, merge_alpha=0.5)
        elif decision.action == "quarantine_candidate":
            q_ids = decision.chosen_slot_ids
            if q_ids:
                self.store.set_slot_state_code(
                    slot_ids=q_ids,
                    state_code=torch.full(
                        (len(q_ids),),
                        slot_state_to_code("quarantined"),
                        device=self.store.slot_values.device,
                        dtype=torch.long,
                    ),
                )
            self._log_write_event(
                action=decision.action,
                request=request,
                input_shape=list(values_2d.shape),
                chosen_slot_ids=[int(x) for x in q_ids],
                reason=decision.reason,
                diagnostics=dict(decision.diagnostics),
            )
            raise RuntimeError(f"write quarantined by arbitrator: {decision.reason}")
        else:
            self._log_write_event(
                action="reject",
                request=request,
                input_shape=list(values_2d.shape),
                chosen_slot_ids=[],
                reason=decision.reason,
                diagnostics=dict(decision.diagnostics),
            )
            raise RuntimeError(f"write rejected by arbitrator: {decision.reason}")

        self._log_write_event(
            action=decision.action,
            request=request,
            input_shape=list(values_2d.shape),
            chosen_slot_ids=[int(x) for x in written_slot_ids],
            reason=decision.reason,
            diagnostics=dict(decision.diagnostics),
        )
        return MemoryWriteOutput(
            written_slot_ids=[int(x) for x in written_slot_ids],
            action=decision.action,
            confidence=float(request.confidence),
            diagnostics={
                "requested_state": request.requested_state,
                "requested_memory_type": request.requested_memory_type,
                "input_shape": list(values_2d.shape),
                "decision_reason": decision.reason,
                "decision_diagnostics": dict(decision.diagnostics),
                "version_counter": int(self.store.version_counter),
                "provenance_logged": True,
            },
        )
