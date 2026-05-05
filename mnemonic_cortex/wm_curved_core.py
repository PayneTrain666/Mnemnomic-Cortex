from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .legacy_enhanced_curved_memory import EnhancedCurvedMemory


class WMCurvedAssociativeCore(nn.Module):
    """Preservation wrapper around EnhancedCurvedMemory.

    WM-1A rules:
    - Do not replace the original curved WM behavior.
    - Delegate to a supplied canonical module when provided.
    - Otherwise instantiate the canonical-compatible EnhancedCurvedMemory.
    - Preserve read/write/process operation surface.
    - Preserve energy-mode hook.
    - Expose trace where available.

    Contract:
    - input: [B,T,D]
    - output: [B,T,D] for read/process
    - output: original x for write
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        mem_slots: int = 7,
        curved_memory: Optional[nn.Module] = None,
        curvature_dim: int = 8,
        spread_steps: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mem_slots = mem_slots
        self.is_external_canonical = curved_memory is not None

        self.curved_memory = curved_memory or EnhancedCurvedMemory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            curvature_dim=curvature_dim,
            mem_slots=mem_slots,
            spread_steps=spread_steps,
        )

    @property
    def last_trace(self) -> Optional[Dict[str, Any]]:
        trace = getattr(self.curved_memory, "last_trace", None)
        if trace is None:
            return None
        if hasattr(trace, "to_dict"):
            return trace.to_dict()
        if isinstance(trace, dict):
            return trace
        return {"trace_repr": repr(trace)}

    def enable_energy_efficient_mode(self, enable: bool = True) -> None:
        if hasattr(self.curved_memory, "enable_energy_efficient_mode"):
            self.curved_memory.enable_energy_efficient_mode(enable)

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        importance: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError("WMCurvedAssociativeCore expects a torch.Tensor")
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x [B,T,{self.input_dim}], got {tuple(x.shape)}")

        # Prefer canonical module's return_trace support when available.
        try:
            return self.curved_memory(
                x,
                operation=operation,
                importance=importance,
                return_trace=return_trace,
            )
        except TypeError:
            out = self.curved_memory(x, operation=operation, importance=importance)
            if return_trace:
                return out, self.last_trace or {"delegated": True, "trace_available": False}
            return out
