from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .ltm_depth_banks import LTMDepthBanks
from .shared_depth_slot_registry import SharedDepthSlotRegistry


class LTMDepthAdapterError(ValueError):
    """Raised when LTM depth adapter input is invalid."""


@dataclass(frozen=True)
class LTMDepthAdapterConfig:
    """Adapter-level configuration for LTM depth routing."""

    enabled: bool = False
    slot_count: int = 2048
    key_dim: int = 256
    value_dim: int = 256
    read_top_k_slots: int = 8
    read_top_k_depths: int = 2
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    @classmethod
    def disabled(cls, key_dim: int = 256, value_dim: Optional[int] = None) -> "LTMDepthAdapterConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(
        cls,
        key_dim: int = 256,
        value_dim: Optional[int] = None,
        slot_count: int = 2048,
    ) -> "LTMDepthAdapterConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "slot_count": self.slot_count,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "read_top_k_slots": self.read_top_k_slots,
            "read_top_k_depths": self.read_top_k_depths,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class LTMDepthAdapter:
    """Optional adapter for LTM depth banks and consolidation proposals."""

    config: LTMDepthAdapterConfig
    banks: Optional[LTMDepthBanks] = None
    registry: Optional[SharedDepthSlotRegistry] = None

    def __post_init__(self) -> None:
        if self.banks is None:
            self.banks = (
                LTMDepthBanks.enabled_default(self.config.key_dim, self.config.value_dim, self.config.slot_count)
                if self.config.enabled
                else LTMDepthBanks.disabled(self.config.key_dim, self.config.value_dim, self.config.slot_count)
            )
        if self.registry is None:
            self.registry = SharedDepthSlotRegistry()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def read_ltm(self, query: torch.Tensor, *, bank_name: str = "cgmn_semantic", return_trace: bool = False):
        bank = self.banks.get(bank_name)
        return bank.read(query, return_trace=return_trace)

    def propose_consolidation(
        self,
        *,
        canonical_slot_id: str,
        content: str,
        bank_name: str,
        slot_index: int,
        depth_index: int,
        value: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        source_stage: str = "REASON-1D",
        source_pack: str = "",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not canonical_slot_id:
            raise LTMDepthAdapterError("canonical_slot_id is required")
        bank = self.banks.get(bank_name)
        bank_result = bank.propose_consolidation_write(
            slot_index=slot_index,
            depth_index=depth_index,
            value=value,
            key=key,
            canonical_slot_id=canonical_slot_id,
        )
        ltm_ref = f"ltm.{bank_name}.slot{int(slot_index)}.z{int(depth_index)}"
        rec = self.registry.create_or_update(
            canonical_slot_id=canonical_slot_id,
            content=content,
            ltm_ref=ltm_ref,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
            depth_roles_present=[int(depth_index)],
            source_stage=source_stage,
            source_pack=source_pack,
            consolidation_status="shadow_proposed",
            registry_lineage=[
                {
                    "event": "ltm_shadow_consolidation_candidate",
                    "bank_name": bank_name,
                    "slot_index": int(slot_index),
                    "depth_index": int(depth_index),
                    "source_stage": source_stage,
                    "source_pack": source_pack,
                }
            ],
        )
        registry_result = self.registry.propose_consolidation(
            canonical_slot_id=canonical_slot_id,
            target_ltm_ref=ltm_ref,
            source_stage=source_stage,
            source_pack=source_pack,
            evidence={"bank_result": bank_result},
        )
        return {
            "committed": False,
            "shadow_only": True,
            "canonical_slot_id": canonical_slot_id,
            "bank_name": bank_name,
            "slot_index": int(slot_index),
            "depth_index": int(depth_index),
            "ltm_ref": ltm_ref,
            "bank_proposal": bank_result,
            "registry_proposal": registry_result,
            "registry_record": rec.to_dict(),
            "safety": {
                "destructive_ltm_replacement": False,
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

    def capacity_metrics(self) -> Dict[str, Any]:
        return self.banks.capacity_metrics()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config": self.config.to_dict(),
            "capacity_metrics": self.capacity_metrics(),
            "registry": self.registry.to_dict(),
            "safety": {
                "destructive_ltm_replacement": False,
                "shared_physical_tensor": False,
                "automatic_memory_store_mutation": False,
                "shadow_consolidation_only_by_default": True,
            },
        }


def ltm_depth_adapter_contract() -> Dict[str, Any]:
    return {
        "module": "ltm_depth_adapter",
        "stage": "REASON-1D",
        "optional": True,
        "default_enabled": False,
        "read_method": "read_ltm(query, bank_name, return_trace)",
        "proposal_method": "propose_consolidation(...)",
        "ltm_depth_routes": {
            "Z0": "core identity / durable identity",
            "Z1": "semantic invariant",
            "Z2": "structural relation",
            "Z3": "contextual binding",
            "Z5": "temporal episode",
            "Z6": "hypothesis / draft abstraction",
            "Z7": "volatile pre-consolidation trace",
        },
        "banks": ["hg_episodic", "cgmn_semantic", "spatial_topological", "procedural_spcp"],
        "shared_physical_tensor": False,
        "shadow_consolidation_only_by_default": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
