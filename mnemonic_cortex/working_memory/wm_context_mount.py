from __future__ import annotations

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
