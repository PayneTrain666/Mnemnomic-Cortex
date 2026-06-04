from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib

import torch

from .ltm_depth_adapter import LTMDepthAdapter
from .mann_depth_adapter import MANNDepthAdapter
from .shared_depth_slot_registry import SharedDepthSlotRegistry

try:
    from mnemonic_cortex.working_memory.context_geometry_maps import build_default_context_geometry_maps
except Exception:  # pragma: no cover - optional fallback for isolated use
    build_default_context_geometry_maps = None


class MANNLTMSharedSlotGeometryError(ValueError):
    """Raised when shared MANN/LTM slot geometry orchestration is invalid."""


GEOMETRY_CHART_GAIN: Dict[str, float] = {
    "euclidean": 1.00,
    "hyperbolic": 1.08,
    "poincare": 1.10,
    "spherical": 0.98,
    "torus": 1.02,
    "complex": 1.04,
    "complex_projective": 1.11,
    "cp_kahler": 1.12,
    "subspace": 1.05,
    "grassmann": 1.06,
    "spatial_se3": 1.07,
    "quaternion": 1.09,
    "dual_quaternion": 1.13,
    "spcp": 1.12,
    "product": 1.06,
    "fiber_bundle": 1.10,
    "tangent_bridge": 1.03,
    "holographic_phase": 1.15,
}


@dataclass(frozen=True)
class SharedGeometrySlotConfig:
    enabled: bool = False
    key_dim: int = 256
    value_dim: int = 256
    default_mann_geometry_map: str = "procedural"
    default_ltm_geometry_map: str = "hierarchical"
    finite_checks: bool = True

    @classmethod
    def disabled(cls, key_dim: int = 256, value_dim: Optional[int] = None) -> "SharedGeometrySlotConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(cls, key_dim: int = 256, value_dim: Optional[int] = None) -> "SharedGeometrySlotConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "default_mann_geometry_map": self.default_mann_geometry_map,
            "default_ltm_geometry_map": self.default_ltm_geometry_map,
            "finite_checks": self.finite_checks,
        }


@dataclass
class MANNLTMSharedSlotGeometry:
    """Coordinate shared MANN/LTM slots and depth-layer chart transforms.

    This module shares canonical slot identity and route metadata across MANN and
    LTM while preserving isolated physical tensors per memory subsystem.
    """

    config: SharedGeometrySlotConfig
    mann_adapter: MANNDepthAdapter
    ltm_adapter: LTMDepthAdapter
    registry: Optional[SharedDepthSlotRegistry] = None

    def __post_init__(self) -> None:
        if self.config.key_dim <= 0 or self.config.value_dim <= 0:
            raise MANNLTMSharedSlotGeometryError("key_dim/value_dim must be positive")
        if self.registry is None:
            self.registry = SharedDepthSlotRegistry()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def run_shared_reasoning(
        self,
        query: torch.Tensor,
        *,
        content: str,
        mann_slot_index: int,
        ltm_slot_index: int,
        hop_id: int = 0,
        ltm_bank_name: str = "cgmn_semantic",
        ltm_depth_index: int = 5,
        mann_geometry_map: Optional[str] = None,
        ltm_geometry_map: Optional[str] = None,
        canonical_slot_id: Optional[str] = None,
        return_trace: bool = False,
    ):
        self._validate_query(query)
        if not self.enabled:
            trace = self._disabled_trace(query)
            if return_trace:
                return query, trace
            return query

        canonical_id = canonical_slot_id or self._canonical_slot_id(content, mann_slot_index, ltm_slot_index)
        mann_map = mann_geometry_map or self.config.default_mann_geometry_map
        ltm_map = ltm_geometry_map or self.config.default_ltm_geometry_map

        mann_out, mann_trace = self.mann_adapter.read_hop(query, hop_id=hop_id, return_trace=True)
        ltm_out, ltm_trace = self.ltm_adapter.read_ltm(mann_out, bank_name=ltm_bank_name, return_trace=True)
        mann_transformed, mann_chart = self._chart_transform(mann_out, geometry_map_name=mann_map, depth_index=hop_id)
        ltm_transformed, ltm_chart = self._chart_transform(ltm_out, geometry_map_name=ltm_map, depth_index=ltm_depth_index)

        fused = 0.5 * (mann_transformed + ltm_transformed)
        if self.config.finite_checks and not torch.isfinite(fused).all():
            raise MANNLTMSharedSlotGeometryError("fused output contains NaN/Inf")

        mann_ref = f"mann.slot{int(mann_slot_index)}.z{int(hop_id)}"
        ltm_ref = f"ltm.{ltm_bank_name}.slot{int(ltm_slot_index)}.z{int(ltm_depth_index)}"
        rec = self.registry.create_or_update(
            canonical_slot_id=canonical_id,
            content=content or "shared_reasoning_slot",
            mann_ref=mann_ref,
            ltm_ref=ltm_ref,
            depth_roles_present=[int(hop_id), int(ltm_depth_index)],
            source_stage="REASON-SHARED-SLOT",
            source_pack="mann_ltm_shared_slot_geometry",
            consolidation_status="shadow_proposed",
            registry_lineage=[
                {
                    "event": "shared_slot_dual_geometry_route",
                    "mann_geometry_map": mann_map,
                    "ltm_geometry_map": ltm_map,
                    "mann_ref": mann_ref,
                    "ltm_ref": ltm_ref,
                }
            ],
        )

        trace = {
            "trace_type": "mann_ltm_shared_slot_geometry",
            "canonical_slot_id": canonical_id,
            "mann_ref": mann_ref,
            "ltm_ref": ltm_ref,
            "mann_geometry_map": mann_map,
            "ltm_geometry_map": ltm_map,
            "mann_chart_transform": mann_chart,
            "ltm_chart_transform": ltm_chart,
            "mann_trace": mann_trace,
            "ltm_trace": ltm_trace,
            "registry_record": rec.to_dict(),
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": True,
                "write_permission_granted": False,
                "shared_physical_tensor": False,
                "chart_transform_applied": True,
            },
        }
        if return_trace:
            return fused, trace
        return fused

    def transform_slot_views(
        self,
        *,
        mann_slot_index: int,
        mann_depth_index: int,
        ltm_bank_name: str,
        ltm_slot_index: int,
        ltm_depth_index: int,
        mann_geometry_map: Optional[str] = None,
        ltm_geometry_map: Optional[str] = None,
    ) -> Dict[str, Any]:
        mann_values = self.mann_adapter.bank.values
        ltm_values = self.ltm_adapter.banks.get(ltm_bank_name).values

        self._validate_slot_depth_indices(
            mann_values=mann_values,
            mann_slot_index=mann_slot_index,
            mann_depth_index=mann_depth_index,
            ltm_values=ltm_values,
            ltm_slot_index=ltm_slot_index,
            ltm_depth_index=ltm_depth_index,
        )

        mann_vec = mann_values[int(mann_slot_index), int(mann_depth_index)]
        ltm_vec = ltm_values[int(ltm_slot_index), int(ltm_depth_index)]

        mann_map = mann_geometry_map or self.config.default_mann_geometry_map
        ltm_map = ltm_geometry_map or self.config.default_ltm_geometry_map
        mann_x, mann_chart = self._chart_transform(mann_vec.unsqueeze(0), geometry_map_name=mann_map, depth_index=mann_depth_index)
        ltm_x, ltm_chart = self._chart_transform(ltm_vec.unsqueeze(0), geometry_map_name=ltm_map, depth_index=ltm_depth_index)

        return {
            "mann_slot_tensor_shape": list(mann_x.shape),
            "ltm_slot_tensor_shape": list(ltm_x.shape),
            "mann_slot_tensor": mann_x.detach().cpu().tolist(),
            "ltm_slot_tensor": ltm_x.detach().cpu().tolist(),
            "mann_chart_transform": mann_chart,
            "ltm_chart_transform": ltm_chart,
            "paamax_metadata": {
                "trace_governance": True,
                "shared_physical_tensor": False,
                "dual_geometry_maps_active": True,
            },
        }

    def _chart_transform(self, tensor: torch.Tensor, *, geometry_map_name: str, depth_index: int):
        geometry = self._geometry_for_depth(geometry_map_name, depth_index)
        gain = GEOMETRY_CHART_GAIN.get(geometry, 1.0)
        depth_scale = 1.0 + 0.04 * float(depth_index + 1)
        transformed = torch.tanh(tensor * (gain * depth_scale)) + 0.1 * torch.sin(tensor * depth_scale)
        if self.config.finite_checks and not torch.isfinite(transformed).all():
            raise MANNLTMSharedSlotGeometryError("chart transform generated NaN/Inf")
        return transformed, {
            "geometry_map": geometry_map_name,
            "depth_index": int(depth_index),
            "geometry": geometry,
            "gain": float(gain),
            "depth_scale": float(depth_scale),
        }

    def _geometry_for_depth(self, geometry_map_name: str, depth_index: int) -> str:
        if not (0 <= int(depth_index) < 8):
            raise MANNLTMSharedSlotGeometryError("depth_index must be in [0,7]")
        depth = int(depth_index)
        if build_default_context_geometry_maps is not None:
            maps = build_default_context_geometry_maps(num_depths=8)
            if geometry_map_name in maps:
                return maps[geometry_map_name].geometry_by_depth[depth]
        fallback = {
            "procedural": ["euclidean", "hyperbolic", "spherical", "torus", "complex", "subspace", "spatial_se3", "spcp"],
            "hierarchical": ["euclidean", "hyperbolic", "poincare", "spherical", "complex_projective", "subspace", "spatial_se3", "spcp"],
            "quantum_holographic": ["euclidean", "hyperbolic", "spherical", "torus", "complex_projective", "holographic_phase", "product", "spcp"],
        }
        return fallback.get(geometry_map_name, fallback["procedural"])[depth]

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise MANNLTMSharedSlotGeometryError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise MANNLTMSharedSlotGeometryError("query must be [B,D] or [B,T,D]")
        if query.size(-1) != self.config.key_dim:
            raise MANNLTMSharedSlotGeometryError(f"query last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise MANNLTMSharedSlotGeometryError("query contains NaN/Inf")

    def _validate_slot_depth_indices(
        self,
        *,
        mann_values: torch.Tensor,
        mann_slot_index: int,
        mann_depth_index: int,
        ltm_values: torch.Tensor,
        ltm_slot_index: int,
        ltm_depth_index: int,
    ) -> None:
        if not (0 <= int(mann_slot_index) < mann_values.size(0)):
            raise MANNLTMSharedSlotGeometryError("mann_slot_index out of range")
        if not (0 <= int(mann_depth_index) < mann_values.size(1)):
            raise MANNLTMSharedSlotGeometryError("mann_depth_index out of range")
        if not (0 <= int(ltm_slot_index) < ltm_values.size(0)):
            raise MANNLTMSharedSlotGeometryError("ltm_slot_index out of range")
        if not (0 <= int(ltm_depth_index) < ltm_values.size(1)):
            raise MANNLTMSharedSlotGeometryError("ltm_depth_index out of range")

    def _canonical_slot_id(self, content: str, mann_slot_index: int, ltm_slot_index: int) -> str:
        payload = f"{content}|mann:{int(mann_slot_index)}|ltm:{int(ltm_slot_index)}"
        return f"mannltm.{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    def _disabled_trace(self, query: torch.Tensor) -> Dict[str, Any]:
        return {
            "trace_type": "mann_ltm_shared_slot_geometry",
            "enabled": False,
            "pass_through": True,
            "input_shape": list(query.shape),
            "output_shape": list(query.shape),
            "paamax_metadata": {
                "trace_governance": True,
                "shared_physical_tensor": False,
                "chart_transform_applied": False,
            },
        }


def mann_ltm_shared_slot_geometry_contract() -> Dict[str, Any]:
    return {
        "module": "mann_ltm_shared_slot_geometry",
        "stage": "REASON-SHARED-SLOT",
        "optional": True,
        "default_enabled": False,
        "uses_shared_depth_slot_registry": True,
        "supports_simultaneous_mann_ltm_geometry_maps": True,
        "chart_transform_over_depth_layers": True,
        "shared_physical_tensor": False,
        "shadow_only_by_default": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
