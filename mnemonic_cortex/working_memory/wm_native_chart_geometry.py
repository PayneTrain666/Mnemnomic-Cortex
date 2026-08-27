"""
Plain-language summary
----------------------
What this file is for: Mount geometry-map charts as native QDT-WM manifolds.
How it fits in the system: Projects depth replicas onto each depth's chart and
exposes map lookup helpers for LTM/MANN cross-attention.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: Mix 0 leaves depth state unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from geometry.chart_native import fit_chart_list, project_with_residual

from .context_geometry_maps import build_default_context_geometry_maps
from .wm_depth_guards import depth_contract_trace


def charts_for_context_map(context_map_name: Optional[str], num_depths: int) -> List[str]:
    maps = build_default_context_geometry_maps(num_depths)
    name = context_map_name or "literal"
    spec = maps.get(name) or maps.get("literal")
    if spec is None:
        return ["euclidean"] * int(num_depths)
    return list(spec.normalized_for_depths(int(num_depths)).geometry_by_depth)


def project_depth_replicas(
    depth_state: torch.Tensor,
    geometry_by_depth: Sequence[str],
    mix: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Project [B,Z,T,3,D] per depth with residual mix. Mix 0 is identity."""
    if depth_state.dim() != 5:
        raise ValueError(f"Expected depth_state [B,Z,T,3,D], got {tuple(depth_state.shape)}")
    z = int(depth_state.size(1))
    charts = fit_chart_list(geometry_by_depth, z)
    mix_v = float(max(0.0, min(1.0, mix)))
    trace: Dict[str, Any] = {
        "trace_type": "wm_native_chart_geometry",
        "enabled": mix_v > 0.0,
        "mix": mix_v,
        "geometry_by_depth": charts,
        "qspin_live_routing": False,
        "shared_slot_writes": False,
        "ltm_writes": False,
        "mann_writes": False,
        "qh_writes": False,
    }
    if mix_v <= 0.0:
        return depth_state, trace
    pieces = []
    geoms = []
    for idx, chart in enumerate(charts):
        projected, meta = project_with_residual(depth_state[:, idx], chart, mix=mix_v)
        pieces.append(projected)
        geoms.append(meta.get("geom"))
    out = torch.stack(pieces, dim=1)
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    trace["geom"] = geoms
    trace["finite"] = bool(torch.isfinite(out).all().item())
    return out, trace


def project_token_chart(
    tokens: torch.Tensor,
    geometry: str,
    mix: float,
) -> torch.Tensor:
    projected, _ = project_with_residual(tokens, geometry, mix=mix)
    return projected


def wm_qd2a_depth_contract() -> dict:
    return depth_contract_trace(module=__name__)
