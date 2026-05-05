from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .context_geometry_maps import ContextGeometryMap, build_default_context_geometry_maps
from .context_map_selector import ContextMapSelector
from .context_triplet_projector import ContextTripletProjector
from .context_depth_adapter import ContextDepthAdapter
from .context_stability_guard import ContextStabilityGuard
from .context_trace import ContextMountTrace


class ContextToWMBridge(nn.Module):
    """Mount context maps onto QDT-WM depth replicas.

    Inputs:
    - context: [B,C,D]
    - depth_state: [B,Z,T,3,D]

    Output:
    - mounted_state: [B,Z,T,3,D]
    - ContextMountTrace
    """

    def __init__(self, dim: int, num_depths: int):
        super().__init__()
        self.dim = dim
        self.num_depths = num_depths
        self.maps = build_default_context_geometry_maps(num_depths)
        self.selector = ContextMapSelector(dim, self.maps)
        self.triplet_projector = ContextTripletProjector(dim)
        self.depth_adapter = ContextDepthAdapter(dim, num_depths)
        self.guard = ContextStabilityGuard()

    def forward(
        self,
        context: torch.Tensor,
        depth_state: torch.Tensor,
        requested_map: Optional[str] = None,
        task_hints: Iterable[str] | None = None,
        paamax_policy_hint: Optional[str] = None,
    ):
        if depth_state.dim() != 5 or depth_state.size(-2) != 3:
            raise ValueError("Expected depth_state [B,Z,T,3,D]")
        selected, selection_trace = self.selector.select(context, requested_map, task_hints, paamax_policy_hint)

        depth_weights = torch.tensor(selected.depth_weights, dtype=depth_state.dtype, device=depth_state.device)
        triplet_bias = torch.tensor(selected.triplet_bias, dtype=depth_state.dtype, device=depth_state.device)

        context_triplet = self.triplet_projector(context)
        depth_context = self.depth_adapter(context_triplet, depth_weights, triplet_bias)
        mount_delta = 0.05 * depth_context.expand(-1, -1, depth_state.size(2), -1, -1)

        report = self.guard.check(context, mount_delta, selected.stability_rules)
        if not report.ok:
            mount_delta = self.guard.repair(mount_delta, selected.stability_rules.get("max_mount_delta", 2.0))
            report = self.guard.check(context, mount_delta, selected.stability_rules)

        mounted = depth_state + mount_delta
        trace = ContextMountTrace(
            selected_map=selected.name,
            selection_reason=selection_trace.reason,
            geometry_by_depth=selected.geometry_by_depth,
            depth_weights=selected.depth_weights,
            mount_strategy=selected.mount_strategy,
            paamax_policy_tags=selected.paamax_policy_tags,
            stability=report.to_dict(),
            selector_scores=selection_trace.scores,
        )
        return mounted, trace
