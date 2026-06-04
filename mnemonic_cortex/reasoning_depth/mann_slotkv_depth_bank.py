from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from .depth_indexed_slot_lattice import DepthIndexedSlotLattice
from .depth_lattice_config import DepthLatticeConfig
from .depth_lattice_types import DepthReadMode, DepthWriteMode, DepthWriteProposal, QHDepthCode


class MANNSlotKVDepthBankError(ValueError):
    """Raised when the MANN SlotKV depth bank receives invalid input."""


@dataclass(frozen=True)
class MANNSlotKVDepthBankConfig:
    """Configuration for a MANN slots × 8 depth key/value bank."""

    enabled: bool = False
    bank_id: str = "mann.slotkv.depth_lattice"
    slot_count: int = 512
    key_dim: int = 256
    value_dim: int = 256
    read_top_k_slots: int = 8
    read_top_k_depths: int = 2
    max_hops: int = 4
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    @classmethod
    def disabled(cls, key_dim: int = 256, value_dim: Optional[int] = None) -> "MANNSlotKVDepthBankConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(
        cls,
        key_dim: int = 256,
        value_dim: Optional[int] = None,
        slot_count: int = 512,
    ) -> "MANNSlotKVDepthBankConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def validate(self) -> None:
        if self.slot_count <= 0:
            raise MANNSlotKVDepthBankError("slot_count must be positive")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise MANNSlotKVDepthBankError("key_dim/value_dim must be positive")
        if not (1 <= self.read_top_k_slots <= self.slot_count):
            raise MANNSlotKVDepthBankError("read_top_k_slots must be within [1, slot_count]")
        if not (1 <= self.read_top_k_depths <= 8):
            raise MANNSlotKVDepthBankError("read_top_k_depths must be within [1, 8]")
        if self.max_hops <= 0:
            raise MANNSlotKVDepthBankError("max_hops must be positive")

    def to_lattice_config(self) -> DepthLatticeConfig:
        self.validate()
        return DepthLatticeConfig(
            bank_id=self.bank_id,
            bank_kind="mann",
            slot_count=self.slot_count,
            num_depths=8,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            read_top_k_slots=self.read_top_k_slots,
            read_top_k_depths=self.read_top_k_depths,
            finite_checks=self.finite_checks,
            no_mutation_by_default=self.no_mutation_by_default,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bank_id": self.bank_id,
            "slot_count": self.slot_count,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "read_top_k_slots": self.read_top_k_slots,
            "read_top_k_depths": self.read_top_k_depths,
            "max_hops": self.max_hops,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class MANNSlotKVDepthBank:
    """MANN SlotKV bank backed by a DepthIndexedSlotLattice.

    This bank is additive. It is not a destructive replacement for any existing
    MANN implementation.
    """

    config: MANNSlotKVDepthBankConfig
    lattice: Optional[DepthIndexedSlotLattice] = None

    def __post_init__(self) -> None:
        self.config.validate()
        if self.lattice is None:
            self.lattice = DepthIndexedSlotLattice(self.config.to_lattice_config())

    @property
    def keys(self) -> torch.Tensor:
        return self.lattice.keys

    @property
    def values(self) -> torch.Tensor:
        return self.lattice.values

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def read(self, query: torch.Tensor, *, hop_id: int = 0, return_trace: bool = False):
        self._validate_query(query)
        if not self.enabled:
            trace = self._disabled_trace(query=query, hop_id=hop_id)
            if return_trace:
                return query, trace
            return query

        value, attention, trace = self.lattice.read(query, read_mode=DepthReadMode.TOP_K, return_trace=True)
        hop_trace = self._augment_hop_trace(trace, hop_id=hop_id)
        if return_trace:
            return value, hop_trace
        return value

    def propose_hop_writes(
        self,
        *,
        slot_index: int,
        value: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        canonical_slot_id: Optional[str] = None,
        hop_id: int = 0,
    ) -> Dict[str, Any]:
        """Create shadow write proposals for MANN reasoning depths.

        Routes:
        - Z4 reasoning_transform
        - Z5 hop_history / temporal_episode
        - Z6 candidate_hypothesis
        - Z7 scratch / volatile_trace
        """

        self._validate_value(value)
        if key is not None:
            self._validate_key(key)
        qh_code = QHDepthCode(
            depth_code="Z4/Z5/Z6/Z7",
            bank_code="MANN_SLOTKV_DEPTH",
            geometry_code="reasoning_depth",
            triplet_code="anchor_direction_phase",
            memory_type_code="mann_reasoning_hop",
            task_mode_code="multi_hop_reasoning",
        )
        routes = {
            "reasoning_transform": 4,
            "hop_history": 5,
            "candidate_hypothesis": 6,
            "scratch_trace": 7,
        }
        proposals = {}
        for name, depth in routes.items():
            proposals[name] = self.lattice.propose_write(
                slot_index=slot_index,
                depth_index=depth,
                value=value,
                key=key,
                mode=DepthWriteMode.SINGLE,
                canonical_slot_id=canonical_slot_id,
                qh_code=qh_code,
            ).to_dict()
        return {
            "committed": False,
            "shadow_only": True,
            "hop_id": hop_id,
            "slot_index": int(slot_index),
            "canonical_slot_id": canonical_slot_id,
            "depth_routes": routes,
            "proposals": proposals,
            "metadata": {
                "bank_id": self.config.bank_id,
                "no_shared_physical_tensor_with_ltm": True,
                "no_permanent_mutation_without_permission": True,
            },
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": True,
                "write_permission_granted": False,
                "no_memory_store_mutation": True,
                "mann_depth_candidate_routing": True,
            },
        }

    def commit_write(
        self,
        proposal: DepthWriteProposal,
        *,
        value: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        allow_mutation: bool = False,
        write_permission: bool = False,
    ) -> Dict[str, Any]:
        return self.lattice.commit_write(
            proposal,
            value=value,
            key=key,
            allow_mutation=allow_mutation,
            write_permission=write_permission,
        )

    def capacity_metrics(self) -> Dict[str, Any]:
        return self.lattice.capacity_metrics()

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise MANNSlotKVDepthBankError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise MANNSlotKVDepthBankError("query must be [B,D] or [B,T,D]")
        if query.size(-1) != self.config.key_dim:
            raise MANNSlotKVDepthBankError(f"query last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise MANNSlotKVDepthBankError("query contains NaN/Inf")

    def _validate_key(self, key: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or key.size(-1) != self.config.key_dim:
            raise MANNSlotKVDepthBankError(f"key last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(key).all():
            raise MANNSlotKVDepthBankError("key contains NaN/Inf")

    def _validate_value(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.size(-1) != self.config.value_dim:
            raise MANNSlotKVDepthBankError(f"value last dim must be {self.config.value_dim}")
        if self.config.finite_checks and not torch.isfinite(value).all():
            raise MANNSlotKVDepthBankError("value contains NaN/Inf")

    def _disabled_trace(self, *, query: torch.Tensor, hop_id: int) -> Dict[str, Any]:
        return {
            "trace_type": "mann_depth_hop_trace",
            "enabled": False,
            "pass_through": True,
            "hop_id": int(hop_id),
            "input_shape": list(query.shape),
            "output_shape": list(query.shape),
            "selected_slots": [],
            "selected_depths": [],
            "depth_entropy": None,
            "support_mass": None,
            "confidence": None,
            "disagreement": None,
            "canonical_slot_ids": [],
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": False,
                "no_memory_store_mutation": True,
            },
        }

    def _augment_hop_trace(self, trace: Dict[str, Any], *, hop_id: int) -> Dict[str, Any]:
        metadata = dict(trace.get("metadata", {}))
        metadata.update({
            "hop_id": int(hop_id),
            "mann_slotkv_depth_bank": True,
            "no_shared_physical_tensor_with_ltm": True,
        })
        return {
            **trace,
            "trace_type": "mann_depth_hop_trace",
            "hop_id": int(hop_id),
            "selected_slots": trace.get("selected_slots", []),
            "selected_depths": trace.get("selected_depths", []),
            "depth_entropy": trace.get("depth_entropy"),
            "support_mass": trace.get("support_mass"),
            "confidence": trace.get("confidence"),
            "disagreement": trace.get("disagreement"),
            "canonical_slot_ids": trace.get("canonical_slot_ids", []),
            "metadata": metadata,
            "paamax_metadata": {
                **trace.get("paamax_metadata", {}),
                "trace_governance": True,
                "write_permission_required": False,
                "no_memory_store_mutation": True,
                "mann_hop_trace": True,
            },
        }


def mann_slotkv_depth_contract() -> Dict[str, Any]:
    return {
        "module": "mann_slotkv_depth_bank",
        "stage": "REASON-1C",
        "optional": True,
        "default_enabled": False,
        "key_shape": "[S,8,K]",
        "value_shape": "[S,8,V]",
        "read_shapes": ["[B,D]", "[B,T,D]"],
        "write_behavior": "shadow proposals only by default",
        "no_shared_physical_tensor_with_ltm": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
