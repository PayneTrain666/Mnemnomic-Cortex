"""
Plain-language summary
----------------------
What this file is for: Opt-in pre-fusion handoff contract. LTM/MANN/SPCP
emit labeled tangent-at-origin messages so dual fusion / chart-fusion policy
can mix them without scraping traces or adding Poincaré/spherical vectors
in ambient space.
How it fits in the system: Linked to the chart-fusion policy testbed. Policy
on implies this handoff on. Off leaves current memory_context-only path.
Status: OPT-IN testbed. Not production.
Important notes: Does not write LTM/MANN/shared slots/QH or activate QSPIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from geometry.chart_native import tangent_at_origin


FUSION_SPACE = "tangent_at_origin"


@dataclass
class PreFusionHandoff:
    """First-class pre-fusion payload for dual fusion.

    tangent is the message at the chart origin (≅ R^D). Euclidean log/exp is
    identity. space must stay tangent_at_origin for chart-fusion policy mixes.
    """

    system: str
    tangent: torch.Tensor
    query_chart: str
    key_charts: List[str] = field(default_factory=list)
    space: str = FUSION_SPACE
    map_name: Optional[str] = None
    weight_shape: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": str(self.system),
            "tangent_shape": list(self.tangent.shape),
            "query_chart": str(self.query_chart),
            "key_charts": list(self.key_charts),
            "space": str(self.space),
            "map_name": self.map_name,
            "weight_shape": list(self.weight_shape or []),
            "handoff": True,
            "qspin_live_routing": False,
        }


def maybe_build_handoff(
    enabled: bool,
    system: str,
    tangent: torch.Tensor,
    stats: Dict[str, Any],
    map_name: Optional[str],
    weights: Optional[torch.Tensor] = None,
) -> Optional[PreFusionHandoff]:
    if not enabled:
        return None
    return PreFusionHandoff(
        system=str(system),
        tangent=tangent,
        query_chart=str(stats.get("query_chart") or "euclidean"),
        key_charts=list(stats.get("key_charts") or []),
        space=FUSION_SPACE,
        map_name=map_name,
        weight_shape=list(weights.shape) if torch.is_tensor(weights) else None,
    )


def wm_token_handoff(tokens: torch.Tensor, map_name: Optional[str] = None) -> PreFusionHandoff:
    """WM source as an explicit Euclidean tangent (identity log-map)."""
    pooled = tokens.mean(dim=1)
    tangent = tangent_at_origin(pooled, "euclidean")
    return PreFusionHandoff(
        system="wm",
        tangent=tangent,
        query_chart="euclidean",
        key_charts=["euclidean"],
        space=FUSION_SPACE,
        map_name=map_name or "wm_tokens",
    )


def tangent_from_output(output: Any, fallback: torch.Tensor) -> Tuple[torch.Tensor, Optional[PreFusionHandoff]]:
    handoff = getattr(output, "handoff", None)
    if isinstance(handoff, PreFusionHandoff) and str(handoff.space) == FUSION_SPACE:
        return handoff.tangent, handoff
    return fallback, None


def charts_from_handoffs(handoffs: Sequence[Optional[PreFusionHandoff]]) -> List[str]:
    charts: List[str] = []
    for handoff in handoffs:
        if handoff is None:
            continue
        if handoff.query_chart:
            charts.append(handoff.query_chart)
        charts.extend(list(handoff.key_charts or []))
    return charts


def prefusion_handoff_contract() -> dict:
    return {
        "trace_type": "wm_prefusion_handoff_contract",
        "module": __name__,
        "opt_in": True,
        "fusion_space": FUSION_SPACE,
        "linked_to_chart_fusion_policy": True,
        "qspin_live_routing": False,
        "shared_slot_writes": False,
        "testbed": True,
    }
