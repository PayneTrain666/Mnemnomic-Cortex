from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curved_slot_state import CurvedSlotStateBank
from .curvature_metric_policy import CurvatureMetricPolicy, CurvatureMetricPolicyOutput
from .context_geometry_maps import build_default_context_geometry_maps, ContextGeometryMap


@dataclass
class DepthSpecificAddressingConfig:
    """Configuration for depth-specific slot addressing.

    Contract:
    - depth_state input: [B,Z,T,3,D]
    - output activation: [B,Z,S]
    """

    dim: int
    num_slots: int
    num_depths: int = 8
    triplet_dim: int = 3
    top_k: int = 4
    content_weight: float = 1.0
    curvature_weight: float = 0.15
    geometry_bias_weight: float = 0.10
    importance_weight: float = 0.15
    confidence_weight: float = 0.15
    temperature: float = 1.0
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass
class DepthSpecificAddressingTrace:
    input_shape: list
    activation_shape: list
    selected_slot_ids_by_depth: List[List[List[str]]]
    top_indices: list
    top_scores: list
    geometry_by_depth: List[str]
    curvature_bias_shape: list
    finite: bool
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepthSpecificAddressingOutput:
    activation: torch.Tensor
    scores: torch.Tensor
    top_indices: torch.Tensor
    top_scores: torch.Tensor
    read_content: torch.Tensor
    trace: DepthSpecificAddressingTrace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_shape": list(self.activation.shape),
            "scores_shape": list(self.scores.shape),
            "read_content_shape": list(self.read_content.shape),
            "top_indices": self.top_indices.detach().cpu().tolist(),
            "top_scores": self.top_scores.detach().cpu().tolist(),
            "trace": self.trace.to_dict(),
        }


GEOMETRY_BIAS_TABLE = {
    "euclidean": 0.00,
    "hyperbolic": 0.05,
    "poincare": 0.06,
    "spherical": 0.03,
    "torus": 0.04,
    "complex": 0.05,
    "complex_projective": 0.07,
    "cp_kahler": 0.07,
    "subspace": 0.06,
    "grassmann": 0.06,
    "spatial_se3": 0.08,
    "quaternion": 0.08,
    "dual_quaternion": 0.09,
    "spcp": 0.08,
    "product": 0.06,
    "fiber_bundle": 0.07,
    "tangent_bridge": 0.05,
    "holographic_phase": 0.09,
}


class DepthSpecificAddressing(nn.Module):
    """Slot addressing per quaternion depth slice.

    This module reads [B,Z,T,3,D] depth state and produces per-depth slot
    activations [B,Z,S]. It uses:
    - per-depth summaries
    - per-depth curvature from CurvatureMetricPolicy
    - geometry_by_depth from a context map
    - CurvedSlotStateBank content/importance/confidence
    """

    def __init__(
        self,
        config: DepthSpecificAddressingConfig,
        slot_bank: CurvedSlotStateBank,
        curvature_policy: Optional[CurvatureMetricPolicy] = None,
        context_maps: Optional[Dict[str, ContextGeometryMap]] = None,
        default_context_map: str = "literal",
    ):
        super().__init__()
        config.validate()
        if slot_bank.config.num_slots != config.num_slots:
            raise ValueError("slot_bank num_slots must match config")
        if slot_bank.config.dim != config.dim:
            raise ValueError("slot_bank dim must match config")
        self.config = config
        self.slot_bank = slot_bank
        self.curvature_policy = curvature_policy
        self.context_maps = context_maps or build_default_context_geometry_maps(config.num_depths)
        self.default_context_map = default_context_map
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
        )
        self.last_trace: Optional[DepthSpecificAddressingTrace] = None

    def _validate_depth_state(self, depth_state: torch.Tensor) -> None:
        if depth_state.dim() != 5:
            raise ValueError(f"Expected depth_state [B,Z,T,3,D], got rank {depth_state.dim()}")
        b, z, t, three, d = depth_state.shape
        if z != self.config.num_depths:
            raise ValueError(f"Expected Z={self.config.num_depths}, got {z}")
        if three != self.config.triplet_dim:
            raise ValueError(f"Expected triplet_dim={self.config.triplet_dim}, got {three}")
        if d != self.config.dim:
            raise ValueError(f"Expected D={self.config.dim}, got {d}")
        if not torch.isfinite(depth_state).all():
            raise ValueError("depth_state contains NaN or Inf")

    def _geometry_by_depth(self, context_map_name: Optional[str]) -> List[str]:
        name = context_map_name or self.default_context_map
        spec = self.context_maps.get(name)
        if spec is None:
            spec = self.context_maps[self.default_context_map]
        return list(spec.geometry_by_depth[: self.config.num_depths])

    def _geometry_bias(self, geometry_by_depth: Sequence[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vals = [GEOMETRY_BIAS_TABLE.get(g, 0.0) for g in geometry_by_depth]
        return torch.tensor(vals, device=device, dtype=dtype).view(1, self.config.num_depths, 1)

    def _curvature_bias(
        self,
        batch_size: int,
        context: Optional[torch.Tensor],
        curvature_output: Optional[CurvatureMetricPolicyOutput],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if curvature_output is None and self.curvature_policy is not None:
            curvature_output = self.curvature_policy(context=context, batch_size=batch_size)
        if curvature_output is None:
            return torch.zeros(batch_size, self.config.num_depths, self.config.num_slots, device=device, dtype=dtype)
        combined = curvature_output.combined_curvature.to(device=device, dtype=dtype)
        if combined.shape != (batch_size, self.config.num_depths, self.config.num_slots):
            raise ValueError(f"curvature combined shape mismatch: {tuple(combined.shape)}")
        return torch.tanh(combined)

    def forward(
        self,
        depth_state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_map_name: Optional[str] = None,
        curvature_output: Optional[CurvatureMetricPolicyOutput] = None,
        return_trace: bool = False,
    ):
        self._validate_depth_state(depth_state)
        b, z, t, three, d = depth_state.shape

        snapshot = self.slot_bank.snapshot()
        content = snapshot.content.to(device=depth_state.device, dtype=depth_state.dtype)
        importance = snapshot.importance.to(device=depth_state.device, dtype=depth_state.dtype)
        confidence = snapshot.confidence.to(device=depth_state.device, dtype=depth_state.dtype)

        # depth query [B,Z,D]
        depth_query = depth_state.mean(dim=(2, 3))
        depth_query = self.query_projection(depth_query)
        q_norm = F.normalize(depth_query, dim=-1, eps=self.config.eps)
        slot_norm = F.normalize(content, dim=-1, eps=self.config.eps)

        content_scores = torch.einsum("bzd,sd->bzs", q_norm, slot_norm)
        curvature_bias = self._curvature_bias(b, context, curvature_output, depth_state.device, depth_state.dtype)
        geometry_by_depth = self._geometry_by_depth(context_map_name)
        geometry_bias = self._geometry_bias(geometry_by_depth, depth_state.device, depth_state.dtype)

        scores = (
            self.config.content_weight * content_scores
            + self.config.curvature_weight * curvature_bias
            + self.config.geometry_bias_weight * geometry_bias
            + self.config.importance_weight * importance.view(1, 1, -1)
            + self.config.confidence_weight * confidence.view(1, 1, -1)
        )
        scores = torch.nan_to_num(scores) / self.config.temperature
        activation = torch.softmax(scores, dim=-1)
        read_content = torch.einsum("bzs,sd->bzd", activation, content)

        k = min(self.config.top_k, self.config.num_slots)
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        idx_list = top_indices.detach().cpu().tolist()
        selected = []
        for batch_rows in idx_list:
            batch_selected = []
            for depth_row in batch_rows:
                batch_selected.append([snapshot.slot_id[int(i)] for i in depth_row])
            selected.append(batch_selected)

        finite = bool(torch.isfinite(activation).all().item() and torch.isfinite(read_content).all().item())
        trace = DepthSpecificAddressingTrace(
            input_shape=list(depth_state.shape),
            activation_shape=list(activation.shape),
            selected_slot_ids_by_depth=selected,
            top_indices=idx_list,
            top_scores=top_scores.detach().cpu().tolist(),
            geometry_by_depth=geometry_by_depth,
            curvature_bias_shape=list(curvature_bias.shape),
            finite=finite,
            paamax_metadata={
                "trace_type": "depth_specific_addressing",
                "context_map_name": context_map_name or self.default_context_map,
                "confidence": 1.0 if finite else 0.0,
                "depth_count": z,
            },
        )
        self.last_trace = trace
        output = DepthSpecificAddressingOutput(
            activation=activation,
            scores=scores,
            top_indices=top_indices,
            top_scores=top_scores,
            read_content=read_content,
            trace=trace,
        )

        if return_trace:
            return output, trace.to_dict()
        return output

    def stability_report(self, depth_state: torch.Tensor) -> Dict[str, Any]:
        out, trace = self.forward(depth_state, return_trace=True)
        return {
            "ok": bool(trace["finite"] and list(out.activation.shape) == [depth_state.size(0), self.config.num_depths, self.config.num_slots]),
            "finite": trace["finite"],
            "activation_shape": list(out.activation.shape),
            "trace": trace,
        }
