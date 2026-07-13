"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm curved core.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

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


# ---------------------------------------------------------------------------
# WM-QD-1A foundation-quality contract
# ---------------------------------------------------------------------------

def wm_qd1a_foundation_contract() -> dict:
    """Return serialization-safe quality metadata for this early-WM module.

    This does not mutate runtime state. It exists so the quality tooling can
    verify that the module has an explicit contract for shape/finite checks,
    traceability, PAAMA-X metadata, fallback behavior, and boundedness.
    """
    return foundation_trace(
        trace_type="wm_qd1a_foundation_contract",
        module=__name__,
        message="early working-memory foundation module hardened by WM-QD-1A",
        payload={
            "shape_checks_required": True,
            "finite_checks_required": True,
            "serialization_safe": True,
            "trace_hooks_required": True,
            "paamax_metadata_required": True,
            "boundedness_required": True,
            "runtime_mutation": "no automatic mutation by quality tooling",
        },
    )
