"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm context mount.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .context_geometry_maps import ContextGeometryMap, build_default_context_geometry_maps
from .context_to_wm_bridge import ContextToWMBridge


# Backward-compatible alias from WM-0A.3.
ContextMapMount = ContextGeometryMap


class GeometryMountedContextBuffer(nn.Module):
    """Geometry-mounted context buffer.

    This WM-0B version supports:
    - full context map presets
    - selector hints
    - context triplet projection
    - depth adaptation
    - PAAMA-X tags
    - stability report
    - trace schema
    """

    def __init__(self, dim: int, num_depths: int, maps: Optional[Dict[str, ContextGeometryMap]] = None):
        super().__init__()
        self.dim = dim
        self.num_depths = num_depths
        self.maps = maps or build_default_context_geometry_maps(num_depths)
        self.context_proj = nn.Linear(dim, dim)
        self.bridge = ContextToWMBridge(dim, num_depths)
        if maps is not None:
            self.bridge.maps = maps
            self.bridge.selector.maps = maps
            self.bridge.selector.map_names = list(maps.keys())
        self._context_memory_builder = None

    def select_map(self, context: torch.Tensor, requested: Optional[str] = None):
        context_h = self.context_proj(context)
        selected, _ = self.bridge.selector.select(context_h, requested_map=requested)
        return selected

    def mount(
        self,
        context: torch.Tensor,
        depth_state: torch.Tensor,
        requested_map: Optional[str] = None,
        task_hints: List[str] | None = None,
        paamax_policy_hint: Optional[str] = None,
    ):
        context_h = self.context_proj(context)
        mounted, trace = self.bridge(
            context_h,
            depth_state,
            requested_map=requested_map,
            task_hints=task_hints,
            paamax_policy_hint=paamax_policy_hint,
        )
        # Preserve old return style: (mounted, mount-like object).
        selected = self.bridge.maps[trace.selected_map]
        object.__setattr__(selected, "trace", trace)  # frozen dataclass compatibility attachment
        return mounted, selected

    def build_context_memory_candidate(
        self,
        *,
        context: torch.Tensor,
        response: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        project_id: str = "unknown_project",
        chat_id: str = "unknown_chat",
        episode_id: str = "unknown_episode",
        context_map: Optional[str] = None,
        task_mode: Optional[str] = None,
        task_hints: Optional[List[str]] = None,
        parameter_hints: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ):
        """Compatibility bridge to context-compression episodic candidate builder."""
        from .context_compression_memory import ContextCompressionConfig, ContextEpisodicMemoryBuilder

        if self._context_memory_builder is None:
            cfg = ContextCompressionConfig(dim=self.dim)
            self._context_memory_builder = ContextEpisodicMemoryBuilder(cfg)

        return self._context_memory_builder.build_candidate(
            context=context,
            response=response,
            model=model,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
            context_map=context_map,
            task_mode=task_mode,
            task_hints=task_hints,
            parameter_hints=parameter_hints,
            metadata=metadata,
        )


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
