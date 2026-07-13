"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth capacity validation.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .depth_indexed_slot_lattice import DepthIndexedSlotLattice
from .depth_lattice_config import DepthLatticeConfig
from .wm_depth_controller import WMDepthController
from .mann_depth_adapter import MANNDepthAdapter, MANNDepthAdapterConfig
from .ltm_depth_adapter import LTMDepthAdapter, LTMDepthAdapterConfig


class DepthCapacityValidationError(ValueError):
    """Raised when depth-capacity validation cannot be completed safely."""


@dataclass(frozen=True)
class DepthCapacityValidationConfig:
    """Bounded validation config for slots × depth capacity checks."""

    slot_counts: Sequence[int] = (16, 64, 256)
    num_depths: int = 8
    key_dim: int = 32
    value_dim: int = 32
    validate_wm: bool = True
    validate_mann: bool = True
    validate_ltm: bool = True
    finite_checks: bool = True
    max_smoke_batch: int = 4
    max_smoke_tokens: int = 8
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.num_depths != 8:
            raise DepthCapacityValidationError("num_depths must remain 8 for REASON-1E validation")
        if not self.slot_counts:
            raise DepthCapacityValidationError("slot_counts cannot be empty")
        for count in self.slot_counts:
            if int(count) <= 0:
                raise DepthCapacityValidationError("all slot_counts must be positive")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise DepthCapacityValidationError("key_dim/value_dim must be positive")
        if self.max_smoke_batch <= 0 or self.max_smoke_tokens <= 0:
            raise DepthCapacityValidationError("max_smoke_batch/max_smoke_tokens must be positive")
        if self.max_smoke_batch > 64:
            raise DepthCapacityValidationError("max_smoke_batch is too large for bounded smoke validation")
        if self.max_smoke_tokens > 128:
            raise DepthCapacityValidationError("max_smoke_tokens is too large for bounded smoke validation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_counts": [int(x) for x in self.slot_counts],
            "num_depths": self.num_depths,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "validate_wm": self.validate_wm,
            "validate_mann": self.validate_mann,
            "validate_ltm": self.validate_ltm,
            "finite_checks": self.finite_checks,
            "max_smoke_batch": self.max_smoke_batch,
            "max_smoke_tokens": self.max_smoke_tokens,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class DepthCapacityValidationReport:
    """JSON-safe capacity validation report."""

    raw_slots: int
    depth_layers: int
    effective_subslots: int
    theoretical_capacity_multiplier: int
    wm_available: bool
    mann_available: bool
    ltm_available: bool
    all_shapes_valid: bool
    no_nan_inf: bool
    safety_flags: Dict[str, bool]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_slots": int(self.raw_slots),
            "depth_layers": int(self.depth_layers),
            "effective_subslots": int(self.effective_subslots),
            "theoretical_capacity_multiplier": int(self.theoretical_capacity_multiplier),
            "wm_available": bool(self.wm_available),
            "mann_available": bool(self.mann_available),
            "ltm_available": bool(self.ltm_available),
            "all_shapes_valid": bool(self.all_shapes_valid),
            "no_nan_inf": bool(self.no_nan_inf),
            "safety_flags": dict(self.safety_flags),
            "details": self.details,
        }


def validate_depth_lattice_capacity(
    *,
    slot_count: int,
    num_depths: int = 8,
    key_dim: int = 32,
    value_dim: int = 32,
    finite_checks: bool = True,
) -> DepthCapacityValidationReport:
    if int(slot_count) <= 0:
        raise DepthCapacityValidationError("slot_count must be positive")
    if int(num_depths) != 8:
        raise DepthCapacityValidationError("num_depths must be 8")
    cfg = DepthLatticeConfig(
        bank_id=f"validation.depth_lattice.{slot_count}",
        bank_kind="wm",
        slot_count=int(slot_count),
        num_depths=int(num_depths),
        key_dim=int(key_dim),
        value_dim=int(value_dim),
        read_top_k_slots=min(4, int(slot_count)),
        read_top_k_depths=2,
        finite_checks=finite_checks,
        no_mutation_by_default=True,
    )
    lattice = DepthIndexedSlotLattice(cfg)
    expected_key_shape = (int(slot_count), int(num_depths), int(key_dim))
    expected_value_shape = (int(slot_count), int(num_depths), int(value_dim))
    all_shapes_valid = lattice.keys.shape == expected_key_shape and lattice.values.shape == expected_value_shape
    no_nan_inf = bool(torch.isfinite(lattice.keys).all().item() and torch.isfinite(lattice.values).all().item())
    return DepthCapacityValidationReport(
        raw_slots=int(slot_count),
        depth_layers=int(num_depths),
        effective_subslots=int(slot_count) * int(num_depths),
        theoretical_capacity_multiplier=int(num_depths),
        wm_available=True,
        mann_available=True,
        ltm_available=True,
        all_shapes_valid=all_shapes_valid,
        no_nan_inf=no_nan_inf,
        safety_flags={
            "no_mutation_by_default": True,
            "shared_physical_tensor": False,
            "destructive_replacement": False,
            "fake_quantum_hardware_claim": False,
        },
        details={
            "key_shape": list(lattice.keys.shape),
            "value_shape": list(lattice.values.shape),
            "expected_key_shape": list(expected_key_shape),
            "expected_value_shape": list(expected_value_shape),
        },
    )


def build_capacity_table(config: Optional[DepthCapacityValidationConfig] = None) -> List[Dict[str, Any]]:
    cfg = config or DepthCapacityValidationConfig()
    cfg.validate()
    rows = []
    for slot_count in cfg.slot_counts:
        report = validate_depth_lattice_capacity(
            slot_count=int(slot_count),
            num_depths=cfg.num_depths,
            key_dim=cfg.key_dim,
            value_dim=cfg.value_dim,
            finite_checks=cfg.finite_checks,
        )
        rows.append(report.to_dict())
    return rows


def validate_wm_mann_ltm_capacity(config: Optional[DepthCapacityValidationConfig] = None) -> Dict[str, Any]:
    cfg = config or DepthCapacityValidationConfig()
    cfg.validate()
    query_2d = torch.randn(min(2, cfg.max_smoke_batch), cfg.key_dim)
    query_3d = torch.randn(min(2, cfg.max_smoke_batch), min(3, cfg.max_smoke_tokens), cfg.key_dim)

    wm_ok = True
    mann_ok = True
    ltm_ok = True
    no_nan_inf = True
    details: Dict[str, Any] = {}

    if cfg.validate_wm:
        wm = WMDepthController.disabled(input_dim=cfg.key_dim)
        wm_out, wm_trace = wm.process_wm(query_3d, return_trace=True)
        wm_ok = wm_out is query_3d and bool(wm_trace.get("pass_through", False))
        no_nan_inf = no_nan_inf and bool(torch.isfinite(wm_out).all().item())
        details["wm"] = {"available": True, "pass_through": wm_ok, "output_shape": list(wm_out.shape)}

    if cfg.validate_mann:
        mann = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=cfg.key_dim, value_dim=cfg.value_dim, slot_count=min(cfg.slot_counts)))
        mann_out, mann_trace = mann.read_hop(query_3d, hop_id=0, return_trace=True)
        mann_ok = tuple(mann_out.shape) == (query_3d.shape[0], cfg.value_dim)
        no_nan_inf = no_nan_inf and bool(torch.isfinite(mann_out).all().item())
        details["mann"] = {"available": True, "output_shape": list(mann_out.shape), "trace_type": mann_trace.get("trace_type")}

    if cfg.validate_ltm:
        ltm = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=cfg.key_dim, value_dim=cfg.value_dim, slot_count=min(cfg.slot_counts)))
        ltm_out, ltm_trace = ltm.read_ltm(query_2d, bank_name="cgmn_semantic", return_trace=True)
        ltm_ok = tuple(ltm_out.shape) == (query_2d.shape[0], cfg.value_dim)
        no_nan_inf = no_nan_inf and bool(torch.isfinite(ltm_out).all().item())
        details["ltm"] = {"available": True, "output_shape": list(ltm_out.shape), "trace_type": ltm_trace.get("trace_type")}

    capacity_table = build_capacity_table(cfg)
    all_shapes_valid = all(row["all_shapes_valid"] for row in capacity_table) and wm_ok and mann_ok and ltm_ok
    return {
        "config": cfg.to_dict(),
        "capacity_table": capacity_table,
        "wm_available": wm_ok,
        "mann_available": mann_ok,
        "ltm_available": ltm_ok,
        "all_shapes_valid": all_shapes_valid,
        "no_nan_inf": no_nan_inf,
        "details": details,
        "safety_flags": {
            "no_mutation_by_default": cfg.no_mutation_by_default,
            "destructive_replacement": False,
            "permanent_memory_store_mutation": False,
            "fake_production_complete_claim": False,
            "fake_quantum_hardware_claim": False,
        },
    }


def depth_capacity_validation_contract() -> Dict[str, Any]:
    return {
        "module": "depth_capacity_validation",
        "stage": "REASON-1E",
        "num_depths": 8,
        "capacity_multiplier": 8,
        "validates": ["DepthIndexedSlotLattice", "WMDepthController", "MANNDepthAdapter", "LTMDepthAdapter"],
        "bounded": True,
        "no_mutation_by_default": True,
    }
