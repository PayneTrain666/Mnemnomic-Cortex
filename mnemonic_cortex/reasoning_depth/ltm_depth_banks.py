"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: ltm depth banks.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .depth_indexed_slot_lattice import DepthIndexedSlotLattice
from .depth_lattice_config import DepthLatticeConfig
from .depth_lattice_types import BankKind, DepthReadMode, DepthWriteMode, QHDepthCode


class LTMDepthBankError(ValueError):
    """Raised when LTM depth bank input is invalid."""


@dataclass(frozen=True)
class LTMDepthBankConfig:
    """Configuration for an individual LTM depth bank."""

    enabled: bool = False
    bank_kind: str = "ltm"
    bank_name: str = "semantic"
    slot_count: int = 2048
    key_dim: int = 256
    value_dim: int = 256
    read_top_k_slots: int = 8
    read_top_k_depths: int = 2
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    @classmethod
    def disabled(cls, bank_name: str = "semantic", key_dim: int = 256, value_dim: Optional[int] = None, slot_count: int = 2048) -> "LTMDepthBankConfig":
        return cls(enabled=False, bank_name=bank_name, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    @classmethod
    def enabled_default(
        cls,
        bank_name: str = "semantic",
        key_dim: int = 256,
        value_dim: Optional[int] = None,
        slot_count: int = 2048,
    ) -> "LTMDepthBankConfig":
        return cls(enabled=True, bank_name=bank_name, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def validate(self) -> None:
        if self.bank_kind != "ltm":
            raise LTMDepthBankError("bank_kind must be 'ltm'")
        if not self.bank_name:
            raise LTMDepthBankError("bank_name is required")
        if self.slot_count <= 0:
            raise LTMDepthBankError("slot_count must be positive")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise LTMDepthBankError("key_dim/value_dim must be positive")
        if not (1 <= self.read_top_k_slots <= self.slot_count):
            raise LTMDepthBankError("read_top_k_slots must be within [1, slot_count]")
        if not (1 <= self.read_top_k_depths <= 8):
            raise LTMDepthBankError("read_top_k_depths must be within [1, 8]")

    def to_lattice_config(self) -> DepthLatticeConfig:
        self.validate()
        return DepthLatticeConfig(
            bank_id=f"ltm.{self.bank_name}.depth_lattice",
            bank_kind=BankKind.LTM,
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
            "bank_kind": self.bank_kind,
            "bank_name": self.bank_name,
            "slot_count": self.slot_count,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "read_top_k_slots": self.read_top_k_slots,
            "read_top_k_depths": self.read_top_k_depths,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class LTMDepthBank:
    """One LTM memory bank backed by a slots × 8 depth lattice."""

    config: LTMDepthBankConfig
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

    def read(self, query: torch.Tensor, *, return_trace: bool = False):
        self._validate_query(query)
        if not self.enabled:
            trace = self._disabled_trace(query)
            if return_trace:
                return query, trace
            return query
        value, attention, trace = self.lattice.read(query, read_mode=DepthReadMode.TOP_K, return_trace=True)
        ltm_trace = self._augment_trace(trace)
        if return_trace:
            return value, ltm_trace
        return value

    def propose_consolidation_write(
        self,
        *,
        slot_index: int,
        depth_index: int,
        value: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        canonical_slot_id: Optional[str] = None,
        memory_type_code: str = "ltm_consolidation_candidate",
    ) -> Dict[str, Any]:
        self._validate_value(value)
        if key is not None:
            self._validate_key(key)
        if not (0 <= int(depth_index) < 8):
            raise LTMDepthBankError("depth_index must be in [0,7]")
        qh_code = QHDepthCode(
            depth_code=f"Z{int(depth_index)}",
            bank_code=f"LTM_{self.config.bank_name.upper()}",
            geometry_code=self._default_geometry_code(),
            triplet_code="anchor_direction_phase",
            memory_type_code=memory_type_code,
            task_mode_code="ltm_depth_consolidation",
        )
        proposal = self.lattice.propose_write(
            slot_index=slot_index,
            depth_index=depth_index,
            value=value,
            key=key,
            mode=DepthWriteMode.SINGLE,
            canonical_slot_id=canonical_slot_id,
            qh_code=qh_code,
        )
        return {
            "committed": False,
            "shadow_only": True,
            "bank_name": self.config.bank_name,
            "slot_index": int(slot_index),
            "depth_index": int(depth_index),
            "canonical_slot_id": canonical_slot_id,
            "proposal": proposal.to_dict(),
            "qh_code": qh_code.to_dict(),
            "safety": {
                "destructive_ltm_replacement": False,
                "permanent_consolidation_requires_gate": True,
                "shared_physical_tensor": False,
            },
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": True,
                "write_permission_granted": False,
                "no_memory_store_mutation": True,
            },
        }

    def capacity_metrics(self) -> Dict[str, Any]:
        metrics = self.lattice.capacity_metrics()
        metrics["bank_name"] = self.config.bank_name
        return metrics

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise LTMDepthBankError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise LTMDepthBankError("query must be [B,D] or [B,T,D]")
        if query.size(-1) != self.config.key_dim:
            raise LTMDepthBankError(f"query last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise LTMDepthBankError("query contains NaN/Inf")

    def _validate_key(self, key: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or key.size(-1) != self.config.key_dim:
            raise LTMDepthBankError(f"key last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(key).all():
            raise LTMDepthBankError("key contains NaN/Inf")

    def _validate_value(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.size(-1) != self.config.value_dim:
            raise LTMDepthBankError(f"value last dim must be {self.config.value_dim}")
        if self.config.finite_checks and not torch.isfinite(value).all():
            raise LTMDepthBankError("value contains NaN/Inf")

    def _disabled_trace(self, query: torch.Tensor) -> Dict[str, Any]:
        return {
            "trace_type": "ltm_depth_trace",
            "enabled": False,
            "pass_through": True,
            "bank_name": self.config.bank_name,
            "input_shape": list(query.shape),
            "output_shape": list(query.shape),
            "selected_slots": [],
            "selected_depths": [],
            "depth_entropy": None,
            "confidence": None,
            "disagreement": None,
            "canonical_slot_ids": [],
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": False,
                "no_memory_store_mutation": True,
            },
        }

    def _augment_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(trace.get("metadata", {}))
        metadata.update({"bank_name": self.config.bank_name, "ltm_depth_bank": True})
        return {
            **trace,
            "trace_type": "ltm_depth_trace",
            "bank_name": self.config.bank_name,
            "selected_slots": trace.get("selected_slots", []),
            "selected_depths": trace.get("selected_depths", []),
            "depth_entropy": trace.get("depth_entropy"),
            "confidence": trace.get("confidence"),
            "disagreement": trace.get("disagreement"),
            "canonical_slot_ids": trace.get("canonical_slot_ids", []),
            "metadata": metadata,
            "paamax_metadata": {
                **trace.get("paamax_metadata", {}),
                "trace_governance": True,
                "write_permission_required": False,
                "no_memory_store_mutation": True,
                "ltm_depth_trace": True,
            },
        }

    def _default_geometry_code(self) -> str:
        return {
            "hg_episodic": "hyperbolic_episodic",
            "cgmn_semantic": "curved_semantic",
            "curved_associative": "curved_associative",
            "spatial_topological": "spatial_topological",
            "procedural_spcp": "spherical_complex_projective",
        }.get(self.config.bank_name, "ltm_depth")


@dataclass
class LTMDepthBanks:
    """Container for canonical LTM depth banks."""

    hg_episodic: LTMDepthBank
    cgmn_semantic: LTMDepthBank
    curved_associative: LTMDepthBank
    spatial_topological: LTMDepthBank
    procedural_spcp: LTMDepthBank

    @classmethod
    def disabled(cls, key_dim: int = 256, value_dim: Optional[int] = None, slot_count: int = 2048) -> "LTMDepthBanks":
        return cls(
            hg_episodic=LTMDepthBank(LTMDepthBankConfig.disabled("hg_episodic", key_dim, value_dim, slot_count)),
            cgmn_semantic=LTMDepthBank(LTMDepthBankConfig.disabled("cgmn_semantic", key_dim, value_dim, slot_count)),
            curved_associative=LTMDepthBank(LTMDepthBankConfig.disabled("curved_associative", key_dim, value_dim, slot_count)),
            spatial_topological=LTMDepthBank(LTMDepthBankConfig.disabled("spatial_topological", key_dim, value_dim, slot_count)),
            procedural_spcp=LTMDepthBank(LTMDepthBankConfig.disabled("procedural_spcp", key_dim, value_dim, slot_count)),
        )

    @classmethod
    def enabled_default(cls, key_dim: int = 256, value_dim: Optional[int] = None, slot_count: int = 2048) -> "LTMDepthBanks":
        return cls(
            hg_episodic=LTMDepthBank(LTMDepthBankConfig.enabled_default("hg_episodic", key_dim, value_dim, slot_count)),
            cgmn_semantic=LTMDepthBank(LTMDepthBankConfig.enabled_default("cgmn_semantic", key_dim, value_dim, slot_count)),
            curved_associative=LTMDepthBank(LTMDepthBankConfig.enabled_default("curved_associative", key_dim, value_dim, slot_count)),
            spatial_topological=LTMDepthBank(LTMDepthBankConfig.enabled_default("spatial_topological", key_dim, value_dim, slot_count)),
            procedural_spcp=LTMDepthBank(LTMDepthBankConfig.enabled_default("procedural_spcp", key_dim, value_dim, slot_count)),
        )

    def get(self, bank_name: str) -> LTMDepthBank:
        bank_name = {
            "hg": "hg_episodic",
            "episodic": "hg_episodic",
            "semantic": "cgmn_semantic",
            "cgmn": "cgmn_semantic",
            "curved": "curved_associative",
            "associative": "curved_associative",
            "spatial": "spatial_topological",
            "spcp": "procedural_spcp",
            "procedural": "procedural_spcp",
        }.get(str(bank_name).strip().lower(), str(bank_name).strip().lower())
        if not hasattr(self, bank_name):
            raise LTMDepthBankError(f"unknown LTM bank: {bank_name}")
        return getattr(self, bank_name)

    def all_banks(self) -> Dict[str, LTMDepthBank]:
        return {
            "hg_episodic": self.hg_episodic,
            "cgmn_semantic": self.cgmn_semantic,
            "curved_associative": self.curved_associative,
            "spatial_topological": self.spatial_topological,
            "procedural_spcp": self.procedural_spcp,
        }

    def capacity_metrics(self) -> Dict[str, Any]:
        return {name: bank.capacity_metrics() for name, bank in self.all_banks().items()}


def ltm_depth_banks_contract() -> Dict[str, Any]:
    return {
        "module": "ltm_depth_banks",
        "stage": "REASON-1D",
        "banks": ["hg_episodic", "cgmn_semantic", "curved_associative", "spatial_topological", "procedural_spcp"],
        "key_shape": "[S,8,K]",
        "value_shape": "[S,8,V]",
        "default_enabled": False,
        "destructive_ltm_replacement": False,
        "shadow_consolidation_only_by_default": True,
        "fake_quantum_hardware_claim": False,
    }
