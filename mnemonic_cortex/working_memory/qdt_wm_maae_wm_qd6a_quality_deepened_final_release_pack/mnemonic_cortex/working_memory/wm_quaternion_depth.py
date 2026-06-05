from __future__ import annotations

from .wm_depth_guards import ensure_depth_state, ensure_token_state, ensure_triplet_axis, normalize_quaternion, ensure_quaternion_pack, depth_contract_trace, assert_depth_compatible_tokens

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions in [...,4] format.

    Quaternion convention:
    - q[..., 0] = w
    - q[..., 1] = x
    - q[..., 2] = y
    - q[..., 3] = z
    """
    if q.size(-1) != 4:
        raise ValueError(f"Expected quaternion trailing dimension 4, got {q.size(-1)}")
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    if q.size(-1) != 4:
        raise ValueError(f"Expected quaternion trailing dimension 4, got {q.size(-1)}")
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions in [...,4] format."""
    if a.size(-1) != 4 or b.size(-1) != 4:
        raise ValueError("Both quaternion tensors must have trailing dimension 4")
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def rotate_vectors_by_quaternion(v: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotate packed 3D vectors by quaternion.

    Inputs:
    - v: [...,3]
    - q: broadcastable to [...,4]

    Output:
    - rotated: [...,3]
    """
    if v.size(-1) != 3:
        raise ValueError(f"Expected vector trailing dimension 3, got {v.size(-1)}")
    q = normalize_quaternion(q, eps=eps)
    zeros = torch.zeros_like(v[..., :1])
    pure_v = torch.cat([zeros, v], dim=-1)
    rotated = quaternion_multiply(quaternion_multiply(q, pure_v), quaternion_conjugate(q))
    return rotated[..., 1:]


@dataclass
class QuaternionDepthTrace:
    input_shape: list
    output_shape: list
    num_depths: int
    triplet_dim: int
    feature_dim: int
    full_3d_blocks: int
    remainder_dim: int
    quaternion_norm_min: float
    quaternion_norm_max: float
    dual_quaternion_hook_available: bool
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuaternionDepthConfig:
    dim: int
    num_depths: int = 8
    triplet_dim: int = 3
    use_trainable_depth_quaternions: bool = True
    preserve_remainder: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3 for anchor/direction/phase")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class QuaternionDepthReplicator(nn.Module):
    """Replicate WM tensors across quaternion dimensional-depth slices.

    WM-2A behavior:
    - input [B,T,D] becomes [B,Z,T,3,D]
    - preserves triplet axis: anchor, direction, phase
    - applies true packed 3D quaternion rotations to feature blocks
    - safely preserves D % 3 remainder dimensions
    - stores trainable per-depth/per-triplet quaternions
    - exposes trace and consistency checks

    This replaces the earlier scalar-only modulation style with real 3D rotation
    across feature blocks while preserving the output contract.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        num_depths: int = 8,
        triplet_dim: int = 3,
        config: Optional[QuaternionDepthConfig] = None,
    ):
        super().__init__()
        if config is None:
            if dim is None:
                raise ValueError("Either dim or config must be provided")
            config = QuaternionDepthConfig(dim=dim, num_depths=num_depths, triplet_dim=triplet_dim)
        config.validate()
        self.config = config
        self.dim = config.dim
        self.num_depths = config.num_depths
        self.triplet_dim = config.triplet_dim
        self.full_3d_blocks = config.dim // 3
        self.remainder_dim = config.dim % 3

        # Identity quaternion initializer [w,x,y,z].
        q = torch.zeros(config.num_depths, config.triplet_dim, 4)
        q[..., 0] = 1.0

        # Add very small deterministic offsets so different depths are not
        # identical after training begins, while remaining near identity.
        if config.use_trainable_depth_quaternions:
            for z in range(config.num_depths):
                for t in range(config.triplet_dim):
                    q[z, t, 1 + ((z + t) % 3)] = 0.001 * float(z + 1) * float(t + 1)
        self.depth_quaternions = nn.Parameter(q, requires_grad=config.use_trainable_depth_quaternions)

        # Trainable affine trim after rotation. Starts as identity-safe.
        self.depth_scale = nn.Parameter(torch.ones(config.num_depths, config.triplet_dim, config.dim))
        self.depth_bias = nn.Parameter(torch.zeros(config.num_depths, config.triplet_dim, config.dim))

        self.last_trace: Optional[QuaternionDepthTrace] = None
        self.dual_quaternion_spatial_hook = None  # Explicit placeholder; not fake-complete.

    def normalized_depth_quaternions(self) -> torch.Tensor:
        return normalize_quaternion(self.depth_quaternions, eps=self.config.eps)

    def _rotate_packed_blocks(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate full 3D feature blocks.

        x: [B,T,D]
        q: [4]
        returns [B,T,D]
        """
        if self.full_3d_blocks == 0:
            return x

        prefix = x[..., : self.full_3d_blocks * 3]
        remainder = x[..., self.full_3d_blocks * 3 :]
        blocks = prefix.reshape(*x.shape[:-1], self.full_3d_blocks, 3)

        # q must broadcast to [B,T,K,4].
        q_expand = q.view(*([1] * (blocks.dim() - 1)), 4).expand(*blocks.shape[:-1], 4)
        rotated_blocks = rotate_vectors_by_quaternion(blocks, q_expand, eps=self.config.eps)
        rotated_prefix = rotated_blocks.reshape(*x.shape[:-1], self.full_3d_blocks * 3)

        if self.remainder_dim and self.config.preserve_remainder:
            return torch.cat([rotated_prefix, remainder], dim=-1)
        return rotated_prefix

    def replicate(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"Expected x [B,T,{self.dim}], got {tuple(x.shape)}")

        q = self.normalized_depth_quaternions()
        depth_triplets = []
        for z in range(self.num_depths):
            triplets = []
            for triplet_idx in range(self.triplet_dim):
                rotated = self._rotate_packed_blocks(x, q[z, triplet_idx])
                scaled = rotated * self.depth_scale[z, triplet_idx].view(1, 1, -1)
                shifted = scaled + self.depth_bias[z, triplet_idx].view(1, 1, -1)
                triplets.append(shifted)
            # [B,T,3,D]
            depth_triplets.append(torch.stack(triplets, dim=2))

        # [B,Z,T,3,D]
        out = torch.stack(depth_triplets, dim=1)

        q_norms = q.norm(dim=-1)
        self.last_trace = QuaternionDepthTrace(
            input_shape=list(x.shape),
            output_shape=list(out.shape),
            num_depths=self.num_depths,
            triplet_dim=self.triplet_dim,
            feature_dim=self.dim,
            full_3d_blocks=self.full_3d_blocks,
            remainder_dim=self.remainder_dim,
            quaternion_norm_min=float(q_norms.detach().min().cpu()),
            quaternion_norm_max=float(q_norms.detach().max().cpu()),
            dual_quaternion_hook_available=self.dual_quaternion_spatial_hook is not None,
            paamax_metadata={
                "trace_type": "quaternion_depth_replication",
                "depth_count": self.num_depths,
                "triplet_axis": ["anchor", "direction", "phase"],
                "remainder_preserved": bool(self.config.preserve_remainder),
                "dual_quaternion_hook_status": "placeholder_not_implemented",
            },
        )
        return out

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        out = self.replicate(x)
        if return_trace:
            return out, self.last_trace.to_dict()
        return out

    def depth_consistency_report(self, x: torch.Tensor) -> Dict[str, Any]:
        out = self.replicate(x)
        finite = bool(torch.isfinite(out).all().item())
        shape_ok = tuple(out.shape) == (x.size(0), self.num_depths, x.size(1), self.triplet_dim, self.dim)
        q = self.normalized_depth_quaternions()
        q_norms = q.norm(dim=-1)
        q_norm_ok = bool(torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5))
        remainder_ok = True
        if self.remainder_dim and self.config.preserve_remainder:
            original_remainder = x[..., -self.remainder_dim :]
            replicated_remainder = out[..., -self.remainder_dim :]
            target = original_remainder.unsqueeze(1).unsqueeze(3).expand_as(replicated_remainder)
            remainder_ok = bool(torch.allclose(replicated_remainder, target, atol=1e-5))
        return {
            "finite": finite,
            "shape_ok": shape_ok,
            "quaternion_norm_ok": q_norm_ok,
            "remainder_ok": remainder_ok,
            "full_3d_blocks": self.full_3d_blocks,
            "remainder_dim": self.remainder_dim,
            "ok": bool(finite and shape_ok and q_norm_ok and remainder_ok),
            "trace": None if self.last_trace is None else self.last_trace.to_dict(),
        }

    @torch.no_grad()
    def set_depth_quaternion(self, depth_index: int, triplet_index: int, quaternion: torch.Tensor) -> None:
        if not 0 <= depth_index < self.num_depths:
            raise IndexError("depth_index out of range")
        if not 0 <= triplet_index < self.triplet_dim:
            raise IndexError("triplet_index out of range")
        q = quaternion.to(device=self.depth_quaternions.device, dtype=self.depth_quaternions.dtype)
        if q.shape != (4,):
            raise ValueError("quaternion must have shape [4]")
        self.depth_quaternions.data[depth_index, triplet_index].copy_(normalize_quaternion(q, eps=self.config.eps))

    def dual_quaternion_status(self) -> Dict[str, Any]:
        return {
            "available": self.dual_quaternion_spatial_hook is not None,
            "status": "placeholder_not_implemented",
            "reason": "WM-2A implements true packed 3D quaternion rotations only; dual-quaternion SE(3) spatial transport is reserved for later spatial/topology integration.",
        }


# ---------------------------------------------------------------------------
# WM-QD-2A quaternion-depth quality contract
# ---------------------------------------------------------------------------

def wm_qd2a_depth_contract() -> dict:
    """Return serialization-safe quality metadata for this depth/assembly module.

    This is a no-mutation contract used by the quality-deepening tooling. It
    declares the expected [B,Z,T,3,D] depth-state invariants, [B,T,D] token-state
    compatibility, quaternion normalization requirement, trace serialization,
    PAAMA-X metadata, fallback behavior, and boundedness expectations.
    """
    return depth_contract_trace(module=__name__)
