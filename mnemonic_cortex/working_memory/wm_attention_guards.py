from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .wm_foundation_guards import (
    WMFoundationValidationError,
    ensure_finite_tensor,
    ensure_rank,
    safe_jsonable,
    foundation_trace,
    bounded_topk,
)
from .wm_depth_guards import ensure_token_state


class WMAttentionValidationError(WMFoundationValidationError):
    """Raised when memory-augmented attention validation fails."""


def _raise_depth(exc: Exception) -> None:
    raise WMAttentionValidationError(str(exc)) from exc


def ensure_attention_query(name: str, tensor: torch.Tensor, expected_dim: Optional[int] = None) -> torch.Tensor:
    """Validate attention query state [B,D] or [B,T,D]."""
    try:
        ensure_finite_tensor(name, tensor)
        if tensor.dim() not in (2, 3):
            raise WMAttentionValidationError(f"{name} must be [B,D] or [B,T,D], got rank {tensor.dim()}")
        if expected_dim is not None and tensor.size(-1) != expected_dim:
            raise WMAttentionValidationError(f"{name} last dim must be {expected_dim}, got {tensor.size(-1)}")
        return tensor
    except WMAttentionValidationError:
        raise
    except Exception as exc:
        _raise_depth(exc)


def ensure_candidate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: Optional[int] = None,
    min_rank: int = 2,
    max_rank: int = 4,
) -> torch.Tensor:
    """Validate candidate tensors such as [B,K,D], [K,D], or [B,L,K,D]."""
    try:
        ensure_finite_tensor(name, tensor)
        if tensor.dim() < min_rank or tensor.dim() > max_rank:
            raise WMAttentionValidationError(f"{name} rank must be in [{min_rank},{max_rank}], got {tensor.dim()}")
        if expected_dim is not None and tensor.size(-1) != expected_dim:
            raise WMAttentionValidationError(f"{name} last dim must be {expected_dim}, got {tensor.size(-1)}")
        return tensor
    except WMAttentionValidationError:
        raise
    except Exception as exc:
        _raise_depth(exc)


def ensure_attention_scores(
    name: str,
    scores: torch.Tensor,
    *,
    min_rank: int = 1,
    max_rank: int = 4,
) -> torch.Tensor:
    """Validate score/logit tensor before softmax or top-k."""
    try:
        ensure_finite_tensor(name, scores)
        if scores.dim() < min_rank or scores.dim() > max_rank:
            raise WMAttentionValidationError(f"{name} rank must be in [{min_rank},{max_rank}], got {scores.dim()}")
        return scores
    except WMAttentionValidationError:
        raise
    except Exception as exc:
        _raise_depth(exc)


def stable_softmax(scores: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Finite guarded softmax with temperature clamp."""
    ensure_attention_scores("scores", scores)
    if temperature <= 0:
        raise WMAttentionValidationError("temperature must be positive")
    temperature = max(float(temperature), 1e-4)
    logits = torch.clamp(scores / temperature, min=-80.0, max=80.0)
    weights = F.softmax(logits, dim=dim)
    ensure_finite_tensor("weights", weights)
    return weights


def bounded_attention_topk(scores: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bounded top-k helper for candidate/lane selection."""
    try:
        ensure_attention_scores("scores", scores)
        return bounded_topk(scores, k=k, dim=dim)
    except Exception as exc:
        raise WMAttentionValidationError(str(exc)) from exc


def ensure_lane_output(
    lane_name: str,
    output: Mapping[str, Any],
    *,
    expected_dim: Optional[int] = None,
    require_trace: bool = True,
) -> Mapping[str, Any]:
    """Validate a retrieval/attention lane output mapping.

    Expected keys are flexible, but if tensor-like values exist they must be finite.
    If `candidates`, `candidate_vectors`, `output`, or `context` exists, it is
    checked as a candidate tensor.
    """
    if not isinstance(output, Mapping):
        raise WMAttentionValidationError(f"{lane_name} output must be a mapping")
    tensor_keys = ["candidates", "candidate_vectors", "output", "context", "values"]
    found_tensor = False
    for key in tensor_keys:
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            found_tensor = True
            ensure_candidate_tensor(f"{lane_name}.{key}", value, expected_dim=expected_dim, min_rank=1, max_rank=5)
    if "scores" in output and isinstance(output["scores"], torch.Tensor):
        ensure_attention_scores(f"{lane_name}.scores", output["scores"])
    if require_trace and "trace" not in output and "metadata" not in output and "paamax_metadata" not in output:
        raise WMAttentionValidationError(f"{lane_name} output must include trace/metadata/paamax_metadata")
    return output


def attention_trace(
    *,
    module: str,
    message: str,
    lane: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    disagreement: float = 0.0,
    conflict_quarantine: bool = False,
    write_permission_required: bool = False,
) -> Dict[str, Any]:
    """Serialization-safe PAAMA-X-compatible attention trace."""
    return foundation_trace(
        trace_type="wm_qd3a_attention_trace",
        module=module,
        message=message,
        payload={
            "lane": lane,
            "candidate_schema_required": True,
            "score_finite_required": True,
            "attention_boundedness_required": True,
            "trace_serialization_required": True,
            "fallback_behavior_required": True,
            **(payload or {}),
        },
        paamax={
            "trace_governance": True,
            "write_permission_required": write_permission_required,
            "conflict_quarantine": conflict_quarantine,
            "confidence": float(confidence),
            "disagreement": float(disagreement),
            "attention_contract": True,
        },
    )


def attention_contract_trace(
    *,
    module: str,
    message: str = "memory-augmented attention module hardened by WM-QD-3A",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a no-mutation contract trace for attention modules."""
    return attention_trace(
        module=module,
        message=message,
        payload={
            "query_shapes": ["[B,D]", "[B,T,D]"],
            "candidate_shapes": ["[K,D]", "[B,K,D]", "[B,L,K,D]"],
            "score_shapes": ["[B,K]", "[B,L,K]", "[K]"],
            "bounded_topk_required": True,
            "stable_softmax_required": True,
            "paamax_metadata_required": True,
            **(payload or {}),
        },
    )


def summarize_attention_tensor(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    ensure_finite_tensor(name, tensor)
    return {
        "name": name,
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "mean_abs": float(tensor.detach().abs().mean().cpu()),
        "max_abs": float(tensor.detach().abs().max().cpu()),
    }


__all__ = [
    "WMAttentionValidationError",
    "ensure_attention_query",
    "ensure_candidate_tensor",
    "ensure_attention_scores",
    "stable_softmax",
    "bounded_attention_topk",
    "ensure_lane_output",
    "attention_trace",
    "attention_contract_trace",
    "summarize_attention_tensor",
]
