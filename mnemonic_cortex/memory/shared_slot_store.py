from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, List

import torch
import torch.nn as nn
from .shared_slot_schema import (
    CODE_TO_SYSTEM,
    SlotMetadata,
    code_to_slot_state,
    slot_state_to_code,
    validate_system_id,
)


class SharedSlotStore(nn.Module):
    """
    Tensor-backed shared slot table with simple metadata and allocation helpers.
    """

    FREE_STATE_CODE = 0

    def __init__(
        self,
        *,
        num_slots: int,
        slot_dim: int,
        num_systems: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive")
        if num_systems <= 0:
            raise ValueError("num_systems must be positive")

        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.num_systems = int(num_systems)
        self.version_counter = 0

        self.register_buffer("slot_values", torch.zeros(num_slots, slot_dim, device=device, dtype=dtype))
        self.register_buffer("slot_confidence", torch.zeros(num_slots, device=device, dtype=torch.float32))
        self.register_buffer("slot_usage", torch.zeros(num_slots, device=device, dtype=torch.float32))
        self.register_buffer("slot_age", torch.zeros(num_slots, device=device, dtype=torch.long))
        self.register_buffer(
            "slot_state_code",
            torch.full((num_slots,), self.FREE_STATE_CODE, device=device, dtype=torch.long),
        )
        self.register_buffer("primary_system_code", torch.zeros(num_slots, device=device, dtype=torch.long))
        self.register_buffer(
            "allowed_read_mask",
            torch.zeros(num_slots, num_systems, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "allowed_write_mask",
            torch.zeros(num_slots, num_systems, device=device, dtype=torch.bool),
        )

        self.metadata: Dict[int, Any] = {}
        self.free_slot_ids: deque[int] = deque(range(self.num_slots))

    def _system_code_for(self, system_id: str) -> int:
        try:
            validate_system_id(system_id)
            from .shared_slot_schema import SYSTEM_TO_CODE

            return int(SYSTEM_TO_CODE[system_id])
        except Exception:
            return 0

    def _sync_metadata_to_tensors(self, slot_id: int, meta: Any) -> None:
        if not isinstance(meta, SlotMetadata):
            return
        sid = int(slot_id)
        self.slot_state_code[sid] = int(slot_state_to_code(meta.state))
        self.slot_confidence[sid] = float(meta.confidence)
        self.slot_usage[sid] = float(meta.usage_score)
        self.slot_age[sid] = int(meta.age_steps)
        self.primary_system_code[sid] = int(self._system_code_for(str(meta.primary_system_id)))

        self.allowed_read_mask[sid] = False
        self.allowed_write_mask[sid] = False
        read_allowed = [x for x in (meta.allowed_read_systems or []) if isinstance(x, str)]
        write_allowed = [x for x in (meta.allowed_write_systems or []) if isinstance(x, str)]
        if read_allowed:
            for sys_id in read_allowed:
                code = self._system_code_for(sys_id)
                if 0 <= code < self.num_systems:
                    self.allowed_read_mask[sid, code] = True
        else:
            # Empty ACL means no explicit restriction.
            self.allowed_read_mask[sid] = True
        if write_allowed:
            for sys_id in write_allowed:
                code = self._system_code_for(sys_id)
                if 0 <= code < self.num_systems:
                    self.allowed_write_mask[sid, code] = True
        else:
            self.allowed_write_mask[sid] = True

    def _normalize_slot_ids(self, slot_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        ids = torch.as_tensor(slot_ids, device=self.slot_values.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return ids
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_slots:
            raise IndexError("slot_ids out of range")
        return ids

    def _coerce_tensor(self, tensor: Any, *, dtype: torch.dtype, shape: tuple[int, ...], name: str) -> torch.Tensor:
        out = torch.as_tensor(tensor, device=self.slot_values.device, dtype=dtype)
        if tuple(out.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(out.shape)}")
        return out

    def _refresh_free_list_for_ids(self, slot_ids: torch.Tensor) -> None:
        free_set = {
            i
            for i in range(self.num_slots)
            if int(self.slot_state_code[i].item()) == self.FREE_STATE_CODE
        }
        self.free_slot_ids = deque(sorted(free_set))

    def _serialize_metadata_item(self, item: Any) -> Any:
        if item is None:
            return None
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Mapping):
            return dict(item)
        return item

    def get_slot_value(self, slot_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        ids = self._normalize_slot_ids(slot_ids)
        return self.slot_values.index_select(0, ids)

    def get_slot_metadata(self, slot_ids: Sequence[int]) -> list[Any]:
        return [self.metadata.get(int(slot_id)) for slot_id in slot_ids]

    def write_slot(
        self,
        slot_ids: torch.Tensor | Sequence[int],
        values: torch.Tensor,
        confidence: torch.Tensor,
        usage: torch.Tensor,
        age: torch.Tensor,
        state_code: torch.Tensor,
        primary_system_code: torch.Tensor,
        allowed_read_mask: torch.Tensor,
        allowed_write_mask: torch.Tensor,
        metadata: Mapping[int, Any] | None = None,
    ) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        k = int(ids.numel())
        if k == 0:
            return

        self.slot_values[ids] = self._coerce_tensor(
            values, dtype=self.slot_values.dtype, shape=(k, self.slot_dim), name="values"
        )
        self.slot_confidence[ids] = self._coerce_tensor(
            confidence, dtype=self.slot_confidence.dtype, shape=(k,), name="confidence"
        )
        self.slot_usage[ids] = self._coerce_tensor(
            usage, dtype=self.slot_usage.dtype, shape=(k,), name="usage"
        )
        self.slot_age[ids] = self._coerce_tensor(age, dtype=self.slot_age.dtype, shape=(k,), name="age")
        self.slot_state_code[ids] = self._coerce_tensor(
            state_code, dtype=self.slot_state_code.dtype, shape=(k,), name="state_code"
        )
        self.primary_system_code[ids] = self._coerce_tensor(
            primary_system_code,
            dtype=self.primary_system_code.dtype,
            shape=(k,),
            name="primary_system_code",
        )
        self.allowed_read_mask[ids] = self._coerce_tensor(
            allowed_read_mask, dtype=torch.bool, shape=(k, self.num_systems), name="allowed_read_mask"
        )
        self.allowed_write_mask[ids] = self._coerce_tensor(
            allowed_write_mask, dtype=torch.bool, shape=(k, self.num_systems), name="allowed_write_mask"
        )

        if metadata:
            for slot_id, meta in metadata.items():
                self.metadata[int(slot_id)] = meta
                self._sync_metadata_to_tensors(int(slot_id), meta)

        self._refresh_free_list_for_ids(ids)
        self.version_counter += 1

    def write_slot_request(self, request: Any) -> None:
        target_ids = getattr(request, "target_slot_ids", None)
        if target_ids is None:
            target_ids = self.allocate_free_slots(1)
        ids = self._normalize_slot_ids(target_ids)
        k = int(ids.numel())
        if k == 0:
            return

        state_code = torch.full((k,), 1, device=self.slot_values.device, dtype=torch.long)
        values = torch.zeros(k, self.slot_dim, device=self.slot_values.device, dtype=self.slot_values.dtype)
        confidence = torch.full((k,), float(getattr(request, "confidence", 0.5)), device=self.slot_values.device)
        usage = torch.zeros(k, device=self.slot_values.device)
        age = torch.zeros(k, device=self.slot_values.device, dtype=torch.long)
        primary = torch.zeros(k, device=self.slot_values.device, dtype=torch.long)
        read_mask = torch.zeros(k, self.num_systems, device=self.slot_values.device, dtype=torch.bool)
        write_mask = torch.zeros(k, self.num_systems, device=self.slot_values.device, dtype=torch.bool)
        req_meta = dict(getattr(request, "extra", {}) or {})
        req_meta["requester_system"] = getattr(request, "requester_system", "unknown")
        metadata = {int(slot_id): req_meta for slot_id in ids.tolist()}

        self.write_slot(ids, values, confidence, usage, age, state_code, primary, read_mask, write_mask, metadata)

    def get_slot_read_result(self, slot_ids: torch.Tensor | Sequence[int]) -> Dict[str, Any]:
        ids = self._normalize_slot_ids(slot_ids)
        values = self.get_slot_value(ids)
        scores = self.slot_confidence.index_select(0, ids).tolist()
        return {
            "slot_ids": ids.tolist(),
            "scores": scores,
            "values_shape": list(values.shape),
            "diagnostics": {"version_counter": self.version_counter},
        }

    def get_slot_write_result(self, slot_ids: torch.Tensor | Sequence[int]) -> Dict[str, Any]:
        ids = self._normalize_slot_ids(slot_ids)
        return {
            "slot_ids": ids.tolist(),
            "version_counter": self.version_counter,
            "state_codes": self.slot_state_code.index_select(0, ids).tolist(),
        }

    def commit_slot_writes(self) -> None:
        self.version_counter += 1

    def commit_slot_write_requests(self) -> None:
        self.version_counter += 1

    def commit_slot_read_results(self) -> None:
        self.version_counter += 1

    def commit_slot_write_results(self) -> None:
        self.version_counter += 1

    def set_slot_value(
        self,
        *,
        slot_ids: torch.Tensor | Sequence[int],
        values: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        self.slot_values[ids] = self._coerce_tensor(
            values, dtype=self.slot_values.dtype, shape=(ids.numel(), self.slot_dim), name="values"
        )
        if confidence is not None:
            self.slot_confidence[ids] = self._coerce_tensor(
                confidence,
                dtype=self.slot_confidence.dtype,
                shape=(ids.numel(),),
                name="confidence",
            )
        self.version_counter += 1

    def set_slot_metadata(self, *, slot_ids: torch.Tensor | Sequence[int], metadata: Mapping[int, Any]) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        for slot_id in ids.tolist():
            if int(slot_id) in metadata:
                item = metadata[int(slot_id)]
                self.metadata[int(slot_id)] = item
                self._sync_metadata_to_tensors(int(slot_id), item)
        self.version_counter += 1

    def set_slot_allowed_read_mask(
        self, *, slot_ids: torch.Tensor | Sequence[int], allowed_read_mask: torch.Tensor
    ) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        self.allowed_read_mask[ids] = self._coerce_tensor(
            allowed_read_mask,
            dtype=torch.bool,
            shape=(ids.numel(), self.num_systems),
            name="allowed_read_mask",
        )
        self.version_counter += 1

    def set_slot_allowed_write_mask(
        self, *, slot_ids: torch.Tensor | Sequence[int], allowed_write_mask: torch.Tensor
    ) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        self.allowed_write_mask[ids] = self._coerce_tensor(
            allowed_write_mask,
            dtype=torch.bool,
            shape=(ids.numel(), self.num_systems),
            name="allowed_write_mask",
        )
        self.version_counter += 1

    def set_slot_free_slot_ids(self, *, slot_ids: torch.Tensor | Sequence[int]) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        self.free_slot_ids = deque(int(i) for i in ids.tolist())
        self.version_counter += 1

    def set_slot_version_counter(self, *, version_counter: int) -> None:
        self.version_counter = int(version_counter)

    def set_slot_primary_system_code(
        self, *, slot_ids: torch.Tensor | Sequence[int], primary_system_code: torch.Tensor
    ) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        self.primary_system_code[ids] = self._coerce_tensor(
            primary_system_code,
            dtype=self.primary_system_code.dtype,
            shape=(ids.numel(),),
            name="primary_system_code",
        )
        self.version_counter += 1

    def set_slot_state_code(self, *, slot_ids: torch.Tensor | Sequence[int], state_code: torch.Tensor) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        codes = self._coerce_tensor(
            state_code, dtype=self.slot_state_code.dtype, shape=(ids.numel(),), name="state_code"
        )
        self.slot_state_code[ids] = codes
        for sid, code in zip(ids.tolist(), codes.tolist()):
            meta = self.metadata.get(int(sid))
            if isinstance(meta, SlotMetadata):
                meta.state = code_to_slot_state(int(code))
        self._refresh_free_list_for_ids(ids)
        self.version_counter += 1

    def allocate_free_slots(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self.free_slot_ids):
            raise RuntimeError(f"requested {count} free slots, only {len(self.free_slot_ids)} available")
        out = [self.free_slot_ids.popleft() for _ in range(count)]
        self.version_counter += 1
        return out

    def deallocate_slots(self, slot_ids: torch.Tensor | Sequence[int]) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        if ids.numel() == 0:
            return

        self.slot_values[ids] = 0
        self.slot_confidence[ids] = 0
        self.slot_usage[ids] = 0
        self.slot_age[ids] = 0
        self.slot_state_code[ids] = self.FREE_STATE_CODE
        self.primary_system_code[ids] = 0
        self.allowed_read_mask[ids] = False
        self.allowed_write_mask[ids] = False

        for slot_id in ids.tolist():
            self.metadata.pop(int(slot_id), None)
        self._refresh_free_list_for_ids(ids)
        self.version_counter += 1

    def increment_usage(self, slot_ids: torch.Tensor | Sequence[int], amount: float = 1.0) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        if ids.numel() == 0:
            return
        self.slot_usage[ids] += float(amount)
        self.version_counter += 1

    def increment_age(self, slot_ids: torch.Tensor | Sequence[int], amount: int = 1) -> None:
        ids = self._normalize_slot_ids(slot_ids)
        if ids.numel() == 0:
            return
        self.slot_age[ids] += int(amount)
        self.version_counter += 1

    # Compatibility helpers for older call sites.
    def get_slot_state(self, slot_id: int) -> str:
        return code_to_slot_state(int(self.slot_state_code[int(slot_id)].item()))

    def get_primary_system(self, slot_id: int) -> Optional[str]:
        code = int(self.primary_system_code[int(slot_id)].item())
        return CODE_TO_SYSTEM.get(code)

    def mark_slot_state(self, *, slot_ids: Sequence[int], state: str) -> None:
        code = slot_state_to_code(state)  # type: ignore[arg-type]
        slot_ids_list = [int(x) for x in slot_ids]
        self.set_slot_state_code(
            slot_ids=slot_ids_list,
            state_code=torch.full(
                (len(slot_ids_list),),
                int(code),
                device=self.slot_values.device,
                dtype=torch.long,
            ),
        )

    def release_slots(self, slot_ids: Sequence[int]) -> None:
        self.deallocate_slots(slot_ids)

    def age_all_slots(self, delta_steps: int = 1) -> None:
        if delta_steps <= 0:
            return
        active = self.active_slot_ids()
        if not active:
            return
        self.increment_age(active, amount=int(delta_steps))

    def active_slot_ids(self) -> List[int]:
        mask = self.slot_state_code != self.FREE_STATE_CODE
        return torch.nonzero(mask, as_tuple=False).reshape(-1).detach().cpu().tolist()

    def summarize(self) -> Dict[str, Any]:
        free_count = sum(1 for x in self.slot_state_code.tolist() if int(x) == self.FREE_STATE_CODE)
        return {
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "num_systems": self.num_systems,
            "version_counter": self.version_counter,
            "free_slots": free_count,
            "used_slots": self.num_slots - free_count,
            "metadata_entries": len(self.metadata),
            "mean_confidence": float(self.slot_confidence.mean().item()),
            "mean_usage": float(self.slot_usage.mean().item()),
            "mean_age": float(self.slot_age.float().mean().item()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.summarize(),
            "free_slot_ids": list(self.free_slot_ids),
            "metadata": {int(k): self._serialize_metadata_item(v) for k, v in self.metadata.items()},
        }

    def trace_summary(self) -> Dict[str, Any]:
        return self.to_dict()