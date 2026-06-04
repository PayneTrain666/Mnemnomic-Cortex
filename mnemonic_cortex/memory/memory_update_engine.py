from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence

import torch

from .memory_write_engine import MemoryWriteEngine, MemoryWriteOutput
from .shared_slot_arbitrator import SharedSlotArbitrator
from .shared_slot_schema import SlotMetadata, SlotWriteRequest
from .shared_slot_store import SharedSlotStore

UpdateMode = Literal["merge", "overwrite", "append_split"]


@dataclass
class MemoryUpdateRequest:
    requester_system: str
    slot_ids: List[int]
    new_values: torch.Tensor
    mode: UpdateMode = "merge"
    confidence: float = 0.5
    reason: str = ""
    requested_state: str = "volatile"
    requested_memory_type: str | None = None
    semantic_tags: List[str] = field(default_factory=list)
    provenance_trace_ids: List[str] = field(default_factory=list)
    extra: Dict[str, object] = field(default_factory=dict)


class MemoryUpdateEngine:
    """
    Update orchestration with split doctrine.

    Uses append_split when:
      - contradiction burden is high
      - semantic content appears incompatible
      - current slot is durable and should not be overwritten
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        write_engine: MemoryWriteEngine,
        arbitrator: SharedSlotArbitrator,
        contradiction_split_threshold: int = 3,
    ) -> None:
        self.store = store
        self.write_engine = write_engine
        self.arbitrator = arbitrator
        self.contradiction_split_threshold = int(max(1, contradiction_split_threshold))
        self._update_trace: List[Dict[str, object]] = []

    def _log_update_event(
        self,
        *,
        action: str,
        request: MemoryUpdateRequest,
        route_reason: str,
        split_reason: str,
        written_slot_ids: List[int],
        diagnostics: Dict[str, object] | None = None,
    ) -> None:
        self._update_trace.append(
            {
                "event_index": len(self._update_trace) + 1,
                "version_counter": int(self.store.version_counter),
                "action": str(action),
                "requester_system": str(request.requester_system),
                "mode": str(request.mode),
                "route_reason": str(route_reason),
                "split_reason": str(split_reason),
                "slot_ids": [int(x) for x in request.slot_ids],
                "written_slot_ids": [int(x) for x in written_slot_ids],
                "input_shape": list(request.new_values.shape),
                "confidence": float(request.confidence),
                "diagnostics": dict(diagnostics or {}),
            }
        )

    def get_update_trace(self) -> List[Dict[str, object]]:
        return [dict(item) for item in self._update_trace]

    def clear_update_trace(self) -> None:
        self._update_trace.clear()

    def _slot_metadata(self, slot_id: int) -> SlotMetadata | None:
        meta = self.store.metadata.get(int(slot_id))
        return meta if isinstance(meta, SlotMetadata) else None

    def _has_high_contradiction_burden(self, slot_ids: Sequence[int]) -> bool:
        for slot_id in slot_ids:
            meta = self._slot_metadata(int(slot_id))
            prov = meta.provenance if meta is not None else None
            burden = int(getattr(prov, "contradiction_count", 0) or 0)
            if burden >= self.contradiction_split_threshold:
                return True
        return False

    def _semantic_incompatible(self, slot_ids: Sequence[int], semantic_tags: Sequence[str]) -> bool:
        incoming = set(str(x) for x in semantic_tags if str(x))
        if not incoming:
            return False
        for slot_id in slot_ids:
            meta = self._slot_metadata(int(slot_id))
            existing = set((meta.semantic_tags if meta is not None else []) or [])
            if existing and incoming.isdisjoint(existing):
                return True
        return False

    def _targets_durable_slot(self, slot_ids: Sequence[int]) -> bool:
        for slot_id in slot_ids:
            meta = self._slot_metadata(int(slot_id))
            if meta is not None and meta.state == "durable":
                return True
        return False

    def _should_append_split(self, request: MemoryUpdateRequest) -> tuple[bool, str]:
        if self._has_high_contradiction_burden(request.slot_ids):
            return True, "high_contradiction_burden"
        if self._semantic_incompatible(request.slot_ids, request.semantic_tags):
            return True, "semantic_incompatible"
        if self._targets_durable_slot(request.slot_ids):
            return True, "durable_slot_protected"
        return False, "normal_update_path"

    def _build_write_request(
        self,
        request: MemoryUpdateRequest,
        *,
        target_slot_ids: List[int] | None,
        split_reason: str,
        force_allocate_new: bool = False,
    ) -> SlotWriteRequest:
        extra = dict(request.extra or {})
        extra["update_reason"] = request.reason
        extra["update_mode"] = request.mode
        extra["split_doctrine_reason"] = split_reason
        extra["force_allocate_new"] = bool(force_allocate_new)
        return SlotWriteRequest(
            requester_system=request.requester_system,
            candidate_value_shape=list(request.new_values.shape),
            target_slot_ids=target_slot_ids,
            requested_state=request.requested_state,  # durable checks remain enforced by write engine doctrine
            requested_memory_type=request.requested_memory_type,
            confidence=float(request.confidence),
            semantic_tags=list(request.semantic_tags),
            provenance_trace_ids=list(request.provenance_trace_ids),
            extra=extra,
        )

    def merge_update(self, request: MemoryUpdateRequest) -> MemoryWriteOutput:
        write_request = self._build_write_request(
            request,
            target_slot_ids=list(request.slot_ids),
            split_reason="merge_update",
        )
        out = self.write_engine.write(request=write_request, values=request.new_values)
        self._log_update_event(
            action="merge",
            request=request,
            route_reason="merge_update",
            split_reason="merge_update",
            written_slot_ids=out.written_slot_ids,
        )
        return out

    def overwrite_update(self, request: MemoryUpdateRequest) -> MemoryWriteOutput:
        write_request = self._build_write_request(
            request,
            target_slot_ids=list(request.slot_ids),
            split_reason="overwrite_update",
        )
        out = self.write_engine.write(request=write_request, values=request.new_values)
        self._log_update_event(
            action="overwrite",
            request=request,
            route_reason="overwrite_update",
            split_reason="overwrite_update",
            written_slot_ids=out.written_slot_ids,
        )
        return out

    def append_split_update(self, request: MemoryUpdateRequest, *, split_reason: str) -> MemoryWriteOutput:
        # Explicitly ask allocator/write path for fresh placement by not providing target slot ids.
        write_request = self._build_write_request(
            request,
            target_slot_ids=None,
            split_reason=split_reason,
            force_allocate_new=True,
        )
        out = self.write_engine.write(request=write_request, values=request.new_values)
        self._log_update_event(
            action="append_split",
            request=request,
            route_reason="append_split_update",
            split_reason=split_reason,
            written_slot_ids=out.written_slot_ids,
        )
        return out

    def update(self, request: MemoryUpdateRequest) -> MemoryWriteOutput:
        if not request.slot_ids:
            return self.append_split_update(request, split_reason="no_existing_targets")

        do_split, split_reason = self._should_append_split(request)
        if request.mode == "append_split" or do_split:
            return self.append_split_update(request, split_reason=split_reason)

        candidate_slot_ids = [int(x) for x in request.slot_ids]
        arb_request = self._build_write_request(
            request,
            target_slot_ids=candidate_slot_ids,
            split_reason="arbitrator_precheck",
        )
        decision = self.arbitrator.decide_write(request=arb_request, candidate_slot_ids=candidate_slot_ids)
        if decision.action == "overwrite":
            return self.overwrite_update(request)
        if decision.action == "merge":
            return self.merge_update(request)
        if decision.action in {"allocate_new", "quarantine_candidate"}:
            return self.append_split_update(request, split_reason=f"arbitrator_{decision.action}")
        return self.append_split_update(request, split_reason="arbitrator_reject")
