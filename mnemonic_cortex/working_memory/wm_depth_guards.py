"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm depth guards.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .wm_foundation_guards import (
    WMFoundationValidationError,
    ensure_finite_tensor,
    ensure_rank,
    ensure_last_dim,
    foundation_trace,
    safe_jsonable,
)


TRIPLET_SIZE = 3


class WMDepthValidationError(WMFoundationValidationError):
    """Raised when quaternion-depth or triplet-depth validation fails."""


def ensure_token_state(name: str, tensor: torch.Tensor, expected_dim: Optional[int] = None) -> torch.Tensor:
    """Validate token state [B,T,D] with depth-specific exception normalization."""
    try:
        ensure_rank(name, tensor, 3)
        if expected_dim is not None:
            ensure_last_dim(name, tensor, expected_dim)
        return tensor
    except Exception as exc:
        raise WMDepthValidationError(str(exc)) from exc


def ensure_depth_state(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_depths: Optional[int] = None,
    expected_dim: Optional[int] = None,
    triplet_size: int = TRIPLET_SIZE,
) -> torch.Tensor:
    """Validate replicated depth state [B,Z,T,3,D].

    Depth-specific validators normalize all failures to WMDepthValidationError
    so callers can catch one depth-layer exception family.
    """
    try:
        ensure_rank(name, tensor, 5)
        if expected_depths is not None and tensor.size(1) != expected_depths:
            raise WMDepthValidationError(f"{name} depth axis Z must be {expected_depths}, got {tensor.size(1)}")
        if tensor.size(3) != triplet_size:
            raise WMDepthValidationError(f"{name} triplet axis must be {triplet_size}, got {tensor.size(3)}")
        if expected_dim is not None and tensor.size(-1) != expected_dim:
            raise WMDepthValidationError(f"{name} last dim must be {expected_dim}, got {tensor.size(-1)}")
        ensure_finite_tensor(name, tensor)
        return tensor
    except WMDepthValidationError:
        raise
    except Exception as exc:
        raise WMDepthValidationError(str(exc)) from exc


def ensure_triplet_axis(name: str, tensor: torch.Tensor, triplet_axis: int = -2, triplet_size: int = TRIPLET_SIZE) -> torch.Tensor:
    """Validate that a tensor contains a 3-lane triplet axis."""
    ensure_finite_tensor(name, tensor)
    if tensor.size(triplet_axis) != triplet_size:
        raise WMDepthValidationError(f"{name} triplet axis {triplet_axis} must be {triplet_size}, got {tensor.size(triplet_axis)}")
    return tensor


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion tensor [...,4] to unit length."""
    ensure_finite_tensor("quaternion", quaternion)
    if quaternion.size(-1) != 4:
        raise WMDepthValidationError(f"quaternion last dim must be 4, got {quaternion.size(-1)}")
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def ensure_unit_quaternion(name: str, quaternion: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """Validate unit quaternion tensor [...,4] with depth-specific exception normalization."""
    try:
        ensure_finite_tensor(name, quaternion)
        if quaternion.size(-1) != 4:
            raise WMDepthValidationError(f"{name} last dim must be 4, got {quaternion.size(-1)}")
        norms = quaternion.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=atol):
            raise WMDepthValidationError(f"{name} must contain unit quaternions")
        return quaternion
    except WMDepthValidationError:
        raise
    except Exception as exc:
        raise WMDepthValidationError(str(exc)) from exc


def ensure_quaternion_pack(
    name: str,
    quaternion: torch.Tensor,
    *,
    expected_depths: Optional[int] = None,
    atol: float = 1e-4,
) -> torch.Tensor:
    """Validate quaternion pack [Z,4] or [...,Z,4]."""
    ensure_unit_quaternion(name, quaternion, atol=atol)
    if expected_depths is not None and quaternion.size(-2) != expected_depths:
        raise WMDepthValidationError(f"{name} depth axis must be {expected_depths}, got {quaternion.size(-2)}")
    return quaternion


def depth_summary(depth_state: torch.Tensor) -> Dict[str, Any]:
    """Return JSON-safe depth-state summary without dumping full tensors."""
    ensure_depth_state("depth_state", depth_state)
    return {
        "shape": list(depth_state.shape),
        "batch": int(depth_state.size(0)),
        "depths": int(depth_state.size(1)),
        "tokens": int(depth_state.size(2)),
        "triplet": int(depth_state.size(3)),
        "dim": int(depth_state.size(4)),
        "finite": bool(torch.isfinite(depth_state).all().item()),
        "mean_abs": float(depth_state.detach().abs().mean().cpu()),
        "max_abs": float(depth_state.detach().abs().max().cpu()),
    }


def token_summary(token_state: torch.Tensor) -> Dict[str, Any]:
    """Return JSON-safe token-state summary without dumping full tensors."""
    ensure_token_state("token_state", token_state)
    return {
        "shape": list(token_state.shape),
        "batch": int(token_state.size(0)),
        "tokens": int(token_state.size(1)),
        "dim": int(token_state.size(2)),
        "finite": bool(torch.isfinite(token_state).all().item()),
        "mean_abs": float(token_state.detach().abs().mean().cpu()),
        "max_abs": float(token_state.detach().abs().max().cpu()),
    }


def depth_contract_trace(
    *,
    module: str,
    message: str = "quaternion depth module hardened by WM-QD-2A",
    expected_depths: int = 8,
    triplet_size: int = TRIPLET_SIZE,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a PAAMA-X-compatible contract trace for depth modules."""
    return foundation_trace(
        trace_type="wm_qd2a_depth_contract",
        module=module,
        message=message,
        payload={
            "depth_state_shape": "[B,Z,T,3,D]",
            "token_state_shape": "[B,T,D]",
            "expected_depths_default": expected_depths,
            "triplet_size": triplet_size,
            "quaternion_normalization_required": True,
            "shape_checks_required": True,
            "finite_checks_required": True,
            "trace_serialization_required": True,
            "fallback_behavior_required": True,
            "boundedness_required": True,
            **(payload or {}),
        },
        paamax={
            "trace_governance": True,
            "write_permission_required": False,
            "conflict_quarantine": False,
            "confidence": 1.0,
            "disagreement": 0.0,
            "depth_contract": True,
        },
    )


def assert_depth_compatible_tokens(depth_state: torch.Tensor, token_state: torch.Tensor) -> None:
    """Validate [B,Z,T,3,D] depth state compatibility with [B,T,D] token state."""
    ensure_depth_state("depth_state", depth_state)
    ensure_token_state("token_state", token_state)
    if depth_state.size(0) != token_state.size(0):
        raise WMDepthValidationError("depth_state and token_state batch sizes differ")
    if depth_state.size(2) != token_state.size(1):
        raise WMDepthValidationError("depth_state and token_state token counts differ")
    if depth_state.size(-1) != token_state.size(-1):
        raise WMDepthValidationError("depth_state and token_state feature dimensions differ")


__all__ = [
    "TRIPLET_SIZE",
    "WMDepthValidationError",
    "ensure_token_state",
    "ensure_depth_state",
    "ensure_triplet_axis",
    "normalize_quaternion",
    "ensure_unit_quaternion",
    "ensure_quaternion_pack",
    "depth_summary",
    "token_summary",
    "depth_contract_trace",
    "assert_depth_compatible_tokens",
]
