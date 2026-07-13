"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: parameter loop adapter.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..parameter_storage_loop_stack import ParameterStorageLoopStack


class ParameterLoopAdapterError(ValueError):
    """Raised when the parameter loop adapter receives invalid input."""


@dataclass(frozen=True)
class ParameterLoopAdapterConfig:
    enabled: bool = False
    key_dim: int = 256
    max_context_tokens: int = 32
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    @classmethod
    def disabled(cls, key_dim: int = 256) -> "ParameterLoopAdapterConfig":
        return cls(enabled=False, key_dim=key_dim)

    @classmethod
    def enabled_default(cls, key_dim: int = 256, max_context_tokens: int = 32) -> "ParameterLoopAdapterConfig":
        return cls(enabled=True, key_dim=key_dim, max_context_tokens=max_context_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "key_dim": int(self.key_dim),
            "max_context_tokens": int(self.max_context_tokens),
            "finite_checks": bool(self.finite_checks),
            "no_mutation_by_default": bool(self.no_mutation_by_default),
        }


@dataclass
class ParameterLoopAdapter:
    """Read-only reasoning-depth adapter over ParameterStorageLoopStack."""

    config: ParameterLoopAdapterConfig
    parameter_loop: Optional[ParameterStorageLoopStack] = None

    def __post_init__(self) -> None:
        if self.config.key_dim <= 0:
            raise ParameterLoopAdapterError("key_dim must be positive")
        if self.config.max_context_tokens <= 0:
            raise ParameterLoopAdapterError("max_context_tokens must be positive")

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.parameter_loop is not None)

    def attach(self, parameter_loop: ParameterStorageLoopStack) -> None:
        self.parameter_loop = parameter_loop

    def read_parameters(self, query: torch.Tensor, *, return_trace: bool = False):
        self._validate_query(query)
        if not self.enabled:
            trace = self._disabled_trace(query)
            if return_trace:
                return query.mean(dim=1) if query.dim() == 3 else query, trace
            return query.mean(dim=1) if query.dim() == 3 else query
        out, trace = self.parameter_loop.read_parameter_summary(
            query,
            top_k=int(self.config.max_context_tokens),
            return_trace=True,
        )
        trace.update(
            {
                "adapter": "parameter_loop_adapter",
                "enabled": True,
                "paamax_metadata": {
                    "trace_governance": True,
                    "write_permission_required": False,
                    "no_memory_store_mutation": True,
                },
            }
        )
        if return_trace:
            return out, trace
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config": self.config.to_dict(),
            "attached": self.parameter_loop is not None,
            "capacity_estimate": self.parameter_loop.estimate_storage_capacity() if self.parameter_loop is not None else None,
            "safety": {
                "read_only": True,
                "shared_slot_write": False,
                "external_ltm_write": False,
                "qspin_runtime_activation": False,
            },
        }

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise ParameterLoopAdapterError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise ParameterLoopAdapterError("query must be [B,D] or [B,T,D]")
        if query.size(-1) != self.config.key_dim:
            raise ParameterLoopAdapterError(f"query last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise ParameterLoopAdapterError("query contains NaN/Inf")

    def _disabled_trace(self, query: torch.Tensor) -> Dict[str, Any]:
        return {
            "trace_type": "parameter_loop_read_only",
            "adapter": "parameter_loop_adapter",
            "enabled": False,
            "pass_through": True,
            "input_shape": list(query.shape),
            "no_memory_store_mutation": True,
        }


def parameter_loop_adapter_contract() -> Dict[str, Any]:
    return {
        "module": "parameter_loop_adapter",
        "stage": "PARAM-LOOP-REASONING-SHADOW",
        "optional": True,
        "default_enabled": False,
        "read_method": "read_parameters(query, return_trace)",
        "read_only": True,
        "shared_slot_write": False,
        "external_ltm_write": False,
        "qspin_runtime_activation": False,
    }


__all__ = [
    "ParameterLoopAdapterConfig",
    "ParameterLoopAdapter",
    "ParameterLoopAdapterError",
    "parameter_loop_adapter_contract",
]
