"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm foundation guards.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import math

import torch


class WMFoundationValidationError(ValueError):
    """Raised when early working-memory foundation validation fails."""


def ensure_finite_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Validate that a tensor contains no NaN/Inf values."""
    if not isinstance(tensor, torch.Tensor):
        raise WMFoundationValidationError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise WMFoundationValidationError(f"{name} contains NaN or Inf")
    return tensor


def ensure_rank(name: str, tensor: torch.Tensor, rank: int) -> torch.Tensor:
    ensure_finite_tensor(name, tensor)
    if tensor.dim() != rank:
        raise WMFoundationValidationError(f"{name} must have rank {rank}, got {tensor.dim()}")
    return tensor


def ensure_last_dim(name: str, tensor: torch.Tensor, dim: int) -> torch.Tensor:
    ensure_finite_tensor(name, tensor)
    if tensor.size(-1) != dim:
        raise WMFoundationValidationError(f"{name} last dim must be {dim}, got {tensor.size(-1)}")
    return tensor


def ensure_shape_prefix(name: str, tensor: torch.Tensor, prefix: Sequence[int]) -> torch.Tensor:
    ensure_finite_tensor(name, tensor)
    if tuple(tensor.shape[: len(prefix)]) != tuple(prefix):
        raise WMFoundationValidationError(
            f"{name} shape prefix must be {tuple(prefix)}, got {tuple(tensor.shape[:len(prefix)])}"
        )
    return tensor


def ensure_probability_vector(name: str, tensor: torch.Tensor, dim: int = -1, atol: float = 1e-4) -> torch.Tensor:
    ensure_finite_tensor(name, tensor)
    if (tensor < -atol).any():
        raise WMFoundationValidationError(f"{name} contains negative probabilities")
    sums = tensor.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol):
        raise WMFoundationValidationError(f"{name} must sum to 1 along dim {dim}")
    return tensor


def clamp_norm(tensor: torch.Tensor, max_norm: float = 1.0e4, eps: float = 1e-8) -> torch.Tensor:
    ensure_finite_tensor("tensor", tensor)
    norm = tensor.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return tensor * scale


def safe_jsonable(value: Any) -> Any:
    """Convert common tensors/dataclasses into JSON-safe structures."""
    if isinstance(value, torch.Tensor):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return {
            "tensor_shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item()),
        }
    if is_dataclass(value):
        return safe_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    return str(value)


def foundation_trace(
    *,
    trace_type: str,
    module: str,
    message: str,
    payload: Optional[Dict[str, Any]] = None,
    paamax: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a serialization-safe early-WM foundation trace payload."""
    return {
        "trace_type": trace_type,
        "module": module,
        "message": message,
        "payload": safe_jsonable(payload or {}),
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": False,
            "conflict_quarantine": False,
            "confidence": 1.0,
            "disagreement": 0.0,
            **(paamax or {}),
        },
    }


def bounded_topk(scores: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    ensure_finite_tensor("scores", scores)
    if k <= 0:
        raise WMFoundationValidationError("k must be positive")
    k = min(k, scores.size(dim))
    return torch.topk(scores, k=k, dim=dim)


def row_stochastic(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ensure_finite_tensor("matrix", matrix)
    matrix = torch.clamp(matrix, min=0.0)
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)


__all__ = [
    "WMFoundationValidationError",
    "ensure_finite_tensor",
    "ensure_rank",
    "ensure_last_dim",
    "ensure_shape_prefix",
    "ensure_probability_vector",
    "clamp_norm",
    "safe_jsonable",
    "foundation_trace",
    "bounded_topk",
    "row_stochastic",
]
