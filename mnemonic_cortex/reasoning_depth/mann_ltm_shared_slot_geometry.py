from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import hashlib

import torch

from geometry.manifold_utils import (
    chart_transform,
    conformal_scale,
    distance,
    gather_curvature,
    geometry_name_to_geom,
    product_et_distance,
    safe_norm,
    symplectic_leapfrog,
    warp_distances,
    wrap_angles,
)

from .ltm_depth_adapter import LTMDepthAdapter
from .mann_depth_adapter import MANNDepthAdapter
from .shared_depth_slot_registry import SharedDepthSlotRegistry

if TYPE_CHECKING:
    from mnemonic_cortex.memory.shared_slot_store import SharedSlotStore
    from mnemonic_cortex.topology_manager import TopologyManagerV3

try:
    from mnemonic_cortex.working_memory.context_geometry_maps import build_default_context_geometry_maps
except Exception:  # pragma: no cover - optional fallback for isolated use
    build_default_context_geometry_maps = None


class MANNLTMSharedSlotGeometryError(ValueError):
    """Raised when shared MANN/LTM slot geometry orchestration is invalid."""


GEOMETRY_CHART_GAIN: Dict[str, float] = {
    "curved": 1.09,
    "curved_associative": 1.09,
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

GEOMETRY_CODE: Dict[str, int] = {
    "euclid": 0,
    "hyperbolic": 1,
    "sphere": 2,
    "torus": 3,
    "cp": 4,
    "grassmann": 5,
    "curved": 1,
}


@dataclass(frozen=True)
class SharedGeometrySlotConfig:
    enabled: bool = False
    key_dim: int = 256
    value_dim: int = 256
    default_mann_geometry_map: str = "procedural"
    default_ltm_geometry_map: str = "hierarchical"
    finite_checks: bool = True
    use_manifold_chart: bool = True
    use_topology_warp: bool = True
    use_symplectic_refine: bool = False
    conformal_b: float = 0.1
    alpha_torus: float = 1.0
    symplectic_step: float = 1e-2

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
            "use_manifold_chart": self.use_manifold_chart,
            "use_topology_warp": self.use_topology_warp,
            "use_symplectic_refine": self.use_symplectic_refine,
            "conformal_b": self.conformal_b,
            "alpha_torus": self.alpha_torus,
            "symplectic_step": self.symplectic_step,
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
    shared_slot_store: Optional["SharedSlotStore"] = None
    topology_manager: Optional["TopologyManagerV3"] = None
    _slot_curvature: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.key_dim <= 0 or self.config.value_dim <= 0:
            raise MANNLTMSharedSlotGeometryError("key_dim/value_dim must be positive")
        if self.registry is None:
            self.registry = SharedDepthSlotRegistry()
        if self._slot_curvature is None:
            slot_count = max(
                int(getattr(self.mann_adapter.config, "slot_count", 512) or 512),
                int(getattr(self.ltm_adapter.config, "slot_count", 512) or 512),
            )
            self._slot_curvature = torch.zeros(slot_count, dtype=torch.float32)

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
        ltm_bank_name = self._normalize_ltm_bank_name(ltm_bank_name)
        mann_map = mann_geometry_map or self.config.default_mann_geometry_map
        ltm_map = ltm_geometry_map or self.config.default_ltm_geometry_map

        mann_out, mann_trace = self.mann_adapter.read_hop(query, hop_id=hop_id, return_trace=True)
        ltm_out, ltm_trace = self.ltm_adapter.read_ltm(mann_out, bank_name=ltm_bank_name, return_trace=True)
        mann_transformed, mann_chart = self._chart_transform(mann_out, geometry_map_name=mann_map, depth_index=hop_id)
        ltm_transformed, ltm_chart = self._chart_transform(ltm_out, geometry_map_name=ltm_map, depth_index=ltm_depth_index)

        mann_transformed, ltm_transformed = self._align_pair_tensors(mann_transformed, ltm_transformed)
        fused = 0.5 * (mann_transformed + ltm_transformed)
        if self.config.use_symplectic_refine and fused.dim() == 2:
            p = torch.zeros_like(fused)
            dH_dq = torch.tanh(fused)
            dH_dp = torch.tanh(p)
            fused, _ = symplectic_leapfrog(fused, p, dH_dq, dH_dp, step=self.config.symplectic_step)

        manifold_diag = self._compute_shared_manifold_diagnostics(
            query=query,
            mann_vec=mann_transformed,
            ltm_vec=ltm_transformed,
            mann_slot_index=mann_slot_index,
            ltm_slot_index=ltm_slot_index,
            mann_geometry=mann_chart.get("geometry", "euclidean"),
            ltm_geometry=ltm_chart.get("geometry", "euclidean"),
        )
        self._sync_shared_slot_store(
            mann_slot_index=mann_slot_index,
            ltm_slot_index=ltm_slot_index,
            mann_geometry=mann_chart.get("geometry", "euclidean"),
            ltm_geometry=ltm_chart.get("geometry", "euclidean"),
            manifold_diag=manifold_diag,
        )

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
            "manifold_diagnostics": manifold_diag,
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
        ltm_bank_name = self._normalize_ltm_bank_name(ltm_bank_name)
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

    def _align_pair_tensors(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if left.shape == right.shape:
            return left, right
        target_dim = max(left.size(-1), right.size(-1))
        return self._resize_last_dim(left, target_dim), self._resize_last_dim(right, target_dim)

    @staticmethod
    def _resize_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        cur = int(tensor.size(-1))
        if cur == target_dim:
            return tensor
        if cur > target_dim:
            return tensor[..., :target_dim]
        pad = target_dim - cur
        return torch.nn.functional.pad(tensor, (0, pad))

    def _chart_transform(self, tensor: torch.Tensor, *, geometry_map_name: str, depth_index: int):
        geometry = self._geometry_for_depth(geometry_map_name, depth_index)
        gain = GEOMETRY_CHART_GAIN.get(geometry, 1.0)
        if self.config.use_manifold_chart:
            transformed, chart_meta = chart_transform(
                tensor,
                geometry=geometry,
                depth_index=depth_index,
                gain=gain,
            )
            meta = {
                "geometry_map": geometry_map_name,
                "depth_index": int(depth_index),
                "geometry": geometry,
                "gain": float(gain),
                "depth_scale": chart_meta["depth_scale"],
                "geom": chart_meta["geom"],
                "manifold_chart": True,
            }
        else:
            depth_scale = 1.0 + 0.04 * float(depth_index + 1)
            transformed = torch.tanh(tensor * (gain * depth_scale)) + 0.1 * torch.sin(tensor * depth_scale)
            meta = {
                "geometry_map": geometry_map_name,
                "depth_index": int(depth_index),
                "geometry": geometry,
                "gain": float(gain),
                "depth_scale": float(depth_scale),
                "manifold_chart": False,
            }
        if self.config.finite_checks and not torch.isfinite(transformed).all():
            raise MANNLTMSharedSlotGeometryError("chart transform generated NaN/Inf")
        return transformed, meta

    def compute_cross_memory_distance(
        self,
        query: torch.Tensor,
        *,
        mann_slot_index: int,
        ltm_slot_index: int,
        ltm_bank_name: str = "cgmn_semantic",
        mann_geometry: str = "euclidean",
        ltm_geometry: str = "euclidean",
    ) -> Dict[str, Any]:
        """Distance between query and shared MANN/LTM slot views with topology warps."""
        self._validate_query(query if query.dim() >= 2 else query.unsqueeze(0))
        q = query.mean(dim=1) if query.dim() == 3 else query
        mann_values = self.mann_adapter.bank.values
        ltm_bank = self.ltm_adapter.banks.get(self._normalize_ltm_bank_name(ltm_bank_name))
        ltm_values = ltm_bank.values if ltm_bank is not None else mann_values

        mann_vec = mann_values[int(mann_slot_index), 0].unsqueeze(0)
        ltm_vec = ltm_values[int(ltm_slot_index), 0].unsqueeze(0)
        mann_x, _ = self._chart_transform(mann_vec, geometry_map_name="procedural", depth_index=0)
        ltm_x, _ = self._chart_transform(ltm_vec, geometry_map_name="hierarchical", depth_index=0)

        mann_geom = geometry_name_to_geom(mann_geometry)
        ltm_geom = geometry_name_to_geom(ltm_geometry)
        d_mann = distance(mann_geom, q, mann_x.expand_as(q)).squeeze(-1)
        d_ltm = distance(ltm_geom, q, ltm_x.expand_as(q)).squeeze(-1)
        d_base = 0.5 * (d_mann + d_ltm)

        if q.size(-1) >= 4 and q.size(0) == mann_x.size(0):
            half = q.size(-1) // 2
            k_e = mann_x[:, :half].unsqueeze(1)
            k_t = wrap_angles(mann_x[:, half:]).unsqueeze(1)
            d_prod = product_et_distance(
                q[:, :half],
                wrap_angles(q[:, half:]),
                k_e,
                k_t,
                alpha=self.config.alpha_torus,
            ).squeeze(-1)
            d_base = 0.5 * d_base + 0.5 * d_prod

        indices = torch.full(
            (q.size(0), 1),
            int(mann_slot_index),
            device=q.device,
            dtype=torch.long,
        )
        curv = self._slot_curvature_for_indices(indices)
        omega = conformal_scale(curv, b=self.config.conformal_b)
        d_warped = warp_distances(d_base.unsqueeze(-1), curvature_idx=curv, conf_scale=omega).squeeze(-1)

        if self.config.use_topology_warp and self.topology_manager is not None:
            subsystem = next(
                (s for s in ("mann_ltm", "hg", "cgmn", "curved") if s in self.topology_manager.subsystems),
                self.topology_manager.subsystems[0],
            )
            d_warped = self.topology_manager.warp_and_blend(
                d_base.unsqueeze(-1),
                indices,
                self._slot_curvature,
                subsystem=subsystem,
                query_feat=q,
                key_feat=mann_values[:, 0, :],
                alpha_torus=self.config.alpha_torus,
                conformal_b=self.config.conformal_b,
            ).squeeze(-1)

        return {
            "distance_mann": float(d_mann.mean().item()),
            "distance_ltm": float(d_ltm.mean().item()),
            "distance_base": float(d_base.mean().item()),
            "distance_warped": float(d_warped.mean().item()),
            "mann_geometry": mann_geometry,
            "ltm_geometry": ltm_geometry,
            "ltm_bank_name": self._normalize_ltm_bank_name(ltm_bank_name),
        }

    def _compute_shared_manifold_diagnostics(
        self,
        *,
        query: torch.Tensor,
        mann_vec: torch.Tensor,
        ltm_vec: torch.Tensor,
        mann_slot_index: int,
        ltm_slot_index: int,
        mann_geometry: str,
        ltm_geometry: str,
    ) -> Dict[str, Any]:
        q = query.mean(dim=1) if query.dim() == 3 else query
        mann_geom = geometry_name_to_geom(mann_geometry)
        ltm_geom = geometry_name_to_geom(ltm_geometry)
        d_mann = distance(mann_geom, q, mann_vec).mean()
        d_ltm = distance(ltm_geom, q, ltm_vec).mean()
        blend = 0.5 * (mann_vec + ltm_vec)
        return {
            "distance_mann_mean": float(d_mann.item()),
            "distance_ltm_mean": float(d_ltm.item()),
            "blend_norm_mean": float(safe_norm(blend, dim=-1).mean().item()),
            "mann_slot_index": int(mann_slot_index),
            "ltm_slot_index": int(ltm_slot_index),
            "mann_geometry": mann_geometry,
            "ltm_geometry": ltm_geometry,
        }

    def _slot_curvature_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        curv = self._slot_curvature.to(indices.device)
        if int(indices.max().item()) >= curv.numel():
            raise MANNLTMSharedSlotGeometryError("slot index exceeds curvature buffer")
        return gather_curvature(curv, indices)

    def _sync_shared_slot_store(
        self,
        *,
        mann_slot_index: int,
        ltm_slot_index: int,
        mann_geometry: str,
        ltm_geometry: str,
        manifold_diag: Dict[str, Any],
    ) -> None:
        if self.shared_slot_store is None:
            return
        store = self.shared_slot_store
        for slot_id, geometry in ((mann_slot_index, mann_geometry), (ltm_slot_index, ltm_geometry)):
            if not (0 <= int(slot_id) < store.num_slots):
                continue
            geom = geometry_name_to_geom(geometry)
            code = GEOMETRY_CODE.get(geom, 0)
            curv_val = 0.5 * (
                float(manifold_diag.get("distance_mann_mean", 0.0))
                + float(manifold_diag.get("distance_ltm_mean", 0.0))
            )
            curv = torch.tanh(torch.tensor([curv_val], dtype=store.slot_curvature.dtype, device=store.slot_curvature.device))
            store.set_slot_curvature(slot_ids=[int(slot_id)], curvature=curv)
            store.set_slot_geometry_code(
                slot_ids=[int(slot_id)],
                geometry_code=torch.tensor([code], device=store.slot_geometry_code.device, dtype=store.slot_geometry_code.dtype),
            )
            if int(slot_id) < self._slot_curvature.numel():
                self._slot_curvature[int(slot_id)] = float(curv.item())
            meta = store.metadata.get(int(slot_id), {}) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta.update(
                {
                    "mann_ltm_geometry": geometry,
                    "manifold_diagnostics": manifold_diag,
                    "shared_slot_geometry": True,
                }
            )
            store.metadata[int(slot_id)] = meta

    def _geometry_for_depth(self, geometry_map_name: str, depth_index: int) -> str:
        if not (0 <= int(depth_index) < 8):
            raise MANNLTMSharedSlotGeometryError("depth_index must be in [0,7]")
        depth = int(depth_index)
        if build_default_context_geometry_maps is not None:
            maps = build_default_context_geometry_maps(num_depths=8)
            if geometry_map_name in maps:
                return maps[geometry_map_name].geometry_by_depth[depth]
        fallback = {
            "curved_associative": ["curved", "hyperbolic", "curved", "euclidean", "curved", "torus", "spherical", "curved"],
            "procedural": ["euclidean", "hyperbolic", "spherical", "torus", "complex", "subspace", "spatial_se3", "spcp"],
            "hierarchical": ["euclidean", "hyperbolic", "poincare", "spherical", "complex_projective", "subspace", "spatial_se3", "spcp"],
            "quantum_holographic": ["euclidean", "hyperbolic", "spherical", "torus", "complex_projective", "holographic_phase", "product", "spcp"],
        }
        return fallback.get(geometry_map_name, fallback["procedural"])[depth]

    def _normalize_ltm_bank_name(self, bank_name: str) -> str:
        normalizer = getattr(self.ltm_adapter, "normalize_bank_name", None)
        if callable(normalizer):
            return str(normalizer(bank_name))
        name = str(bank_name).strip().lower()
        return {
            "hg": "hg_episodic",
            "episodic": "hg_episodic",
            "semantic": "cgmn_semantic",
            "cgmn": "cgmn_semantic",
            "curved": "curved_associative",
            "associative": "curved_associative",
            "spatial": "spatial_topological",
            "spcp": "procedural_spcp",
            "procedural": "procedural_spcp",
        }.get(name, name)

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
        "uses_geometry_manifold_utils": True,
        "supports_shared_slot_store_curvature": True,
        "supports_topology_manager_warp": True,
        "ltm_banks": ["hg_episodic", "cgmn_semantic", "curved_associative", "spatial_topological", "procedural_spcp"],
        "shared_physical_tensor": False,
        "shadow_only_by_default": True,
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
