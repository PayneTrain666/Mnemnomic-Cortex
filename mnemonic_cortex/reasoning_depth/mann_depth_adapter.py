"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: mann depth adapter.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .mann_slotkv_depth_bank import MANNSlotKVDepthBank, MANNSlotKVDepthBankConfig, MANNSlotKVDepthBankError


class MANNDepthAdapterError(ValueError):
    """Raised when the MANN depth adapter cannot route safely."""


@dataclass(frozen=True)
class MANNDepthAdapterConfig:
    """Adapter-level config for optional MANN depth routing."""

    enabled: bool = False
    slot_count: int = 512
    key_dim: int = 256
    value_dim: int = 256
    read_top_k_slots: int = 8
    read_top_k_depths: int = 2
    max_hops: int = 4
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    @classmethod
    def disabled(cls, key_dim: int = 256, value_dim: Optional[int] = None) -> "MANNDepthAdapterConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(
        cls,
        key_dim: int = 256,
        value_dim: Optional[int] = None,
        slot_count: int = 512,
    ) -> "MANNDepthAdapterConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def to_bank_config(self) -> MANNSlotKVDepthBankConfig:
        return MANNSlotKVDepthBankConfig(
            enabled=self.enabled,
            slot_count=self.slot_count,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            read_top_k_slots=self.read_top_k_slots,
            read_top_k_depths=self.read_top_k_depths,
            max_hops=self.max_hops,
            finite_checks=self.finite_checks,
            no_mutation_by_default=self.no_mutation_by_default,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
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
class MANNDepthAdapter:
    """Optional depth adapter for MANN reasoning scratchpad.

    The adapter wraps a MANNSlotKVDepthBank and exposes hop-oriented read/write
    proposal helpers. It does not replace any existing MANN runtime.
    """

    config: MANNDepthAdapterConfig
    bank: Optional[MANNSlotKVDepthBank] = None

    def __post_init__(self) -> None:
        if self.config.max_hops <= 0:
            raise MANNDepthAdapterError("max_hops must be positive")
        if self.bank is None:
            self.bank = MANNSlotKVDepthBank(self.config.to_bank_config())

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def read_hop(self, query: torch.Tensor, *, hop_id: int = 0, return_trace: bool = False):
        if not (0 <= int(hop_id) < self.config.max_hops):
            raise MANNDepthAdapterError("hop_id outside configured max_hops")
        return self.bank.read(query, hop_id=hop_id, return_trace=return_trace)

    def propose_hop_memory(
        self,
        *,
        slot_index: int,
        value: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        canonical_slot_id: Optional[str] = None,
        hop_id: int = 0,
    ) -> Dict[str, Any]:
        if not (0 <= int(hop_id) < self.config.max_hops):
            raise MANNDepthAdapterError("hop_id outside configured max_hops")
        return self.bank.propose_hop_writes(
            slot_index=slot_index,
            value=value,
            key=key,
            canonical_slot_id=canonical_slot_id,
            hop_id=hop_id,
        )

    def capacity_metrics(self) -> Dict[str, Any]:
        return self.bank.capacity_metrics()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config": self.config.to_dict(),
            "capacity_metrics": self.capacity_metrics(),
            "safety": {
                "destructive_mann_replacement": False,
                "shared_physical_tensor_with_ltm": False,
                "automatic_memory_store_mutation": False,
                "shadow_writes_only_by_default": True,
            },
        }


def mann_depth_contract() -> Dict[str, Any]:
    return {
        "module": "mann_depth_adapter",
        "stage": "REASON-1C",
        "optional": True,
        "default_enabled": False,
        "read_method": "read_hop(query, hop_id, return_trace)",
        "write_method": "propose_hop_memory(...)",
        "hop_trace_fields": [
            "hop_id",
            "selected_slots",
            "selected_depths",
            "depth_entropy",
            "support_mass",
            "confidence",
            "disagreement",
            "canonical_slot_ids",
        ],
        "write_depth_routes": {
            "Z4": "reasoning_transform",
            "Z5": "hop_history / temporal_episode",
            "Z6": "candidate_hypothesis",
            "Z7": "scratch / volatile_trace",
        },
        "no_shared_physical_tensor_with_ltm": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
