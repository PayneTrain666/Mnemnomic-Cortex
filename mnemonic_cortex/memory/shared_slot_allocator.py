from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from .shared_slot_schema import SlotMetadata, SlotState, code_to_slot_state, slot_state_to_code
from .shared_slot_store import SharedSlotStore


class SharedSlotAllocator:
    """
    Allocation policy helper for `SharedSlotStore`.
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        reuse_threshold: float = 0.20,
        pressure_threshold: float = 0.90,
    ) -> None:
        self.store = store
        self.reuse_threshold = float(reuse_threshold)
        self.pressure_threshold = float(pressure_threshold)

    def pressure_score(self) -> float:
        return 1.0 - (len(self.store.free_slot_ids) / max(1, self.store.num_slots))

    def candidate_eviction_slots(
        self,
        *,
        count: int,
        exclude_states: Tuple[SlotState, ...] = ("durable", "quarantined"),
    ) -> List[int]:
        if count <= 0:
            return []
        exclude_codes = {slot_state_to_code(s) for s in exclude_states}

        scored: List[tuple[float, int]] = []
        for idx in range(self.store.num_slots):
            state_code = int(self.store.slot_state_code[idx].item())
            if state_code in exclude_codes:
                continue
            # Lower usage/confidence and higher age are easiest to evict.
            usage = float(self.store.slot_usage[idx].item())
            confidence = float(self.store.slot_confidence[idx].item())
            age = float(self.store.slot_age[idx].item())
            score = (1.0 - confidence) + (1.0 / (1.0 + usage)) + (0.01 * age)
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [slot_id for _, slot_id in scored[:count]]

    def _mark_allocated(self, slot_ids: Sequence[int], requester_system: str, requested_state: SlotState) -> None:
        if not slot_ids:
            return
        ids = torch.tensor(list(slot_ids), device=self.store.slot_values.device, dtype=torch.long)
        state_code = torch.full(
            (len(slot_ids),),
            slot_state_to_code(requested_state),
            device=self.store.slot_values.device,
            dtype=torch.long,
        )
        self.store.set_slot_state_code(slot_ids=ids, state_code=state_code)

        meta = {}
        for slot_id in slot_ids:
            prior = self.store.metadata.get(int(slot_id))
            if isinstance(prior, SlotMetadata):
                m = prior
                m.state = requested_state
                m.primary_system_id = requester_system
            else:
                m = SlotMetadata(slot_id=int(slot_id), state=requested_state, primary_system_id=requester_system)
            meta[int(slot_id)] = m
        self.store.set_slot_metadata(slot_ids=ids, metadata=meta)

    def allocate(
        self,
        *,
        count: int,
        requester_system: str,
        requested_state: SlotState = "volatile",
    ) -> List[int]:
        if count <= 0:
            return []

        allocated: List[int] = []
        free_available = len(self.store.free_slot_ids)
        if free_available > 0:
            take = min(count, free_available)
            allocated.extend(self.store.allocate_free_slots(take))

        deficit = count - len(allocated)
        if deficit > 0:
            evict = self.candidate_eviction_slots(count=deficit)
            if len(evict) < deficit:
                raise RuntimeError(f"insufficient capacity: requested={count}, allocated={len(allocated)}")
            self.store.deallocate_slots(torch.tensor(evict, dtype=torch.long))
            allocated.extend(self.store.allocate_free_slots(deficit))

        self._mark_allocated(allocated, requester_system, requested_state)
        return allocated

    def allocate_slots(
        self,
        *,
        count: int,
        requester_system: str,
        requested_state: SlotState = "volatile",
    ) -> List[int]:
        return self.allocate(count=count, requester_system=requester_system, requested_state=requested_state)

    def commit_slot_write_requests(self) -> None:
        self.store.commit_slot_write_requests()

    def commit_slot_read_results(self) -> None:
        self.store.commit_slot_read_results()

    def get_slot_value(self, slot_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return self.store.get_slot_value(slot_ids)

    def get_slot_metadata(self, slot_ids: List[int]) -> List[SlotMetadata | Any]:
        return self.store.get_slot_metadata(slot_ids)

    def summarize(self) -> Dict[str, Any]:
        return {
            "pressure_score": self.pressure_score(),
            "reuse_threshold": self.reuse_threshold,
            "pressure_threshold": self.pressure_threshold,
            "store": self.store.summarize(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summarize()

    def trace_summary(self) -> Dict[str, Any]:
        data = self.summarize()
        data["eviction_candidate_preview"] = self.candidate_eviction_slots(count=8)
        data["state_histogram"] = {
            code_to_slot_state(code): int((self.store.slot_state_code == code).sum().item())
            for code in range(6)
        }
        return data