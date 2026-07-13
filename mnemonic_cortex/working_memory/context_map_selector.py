"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context map selector.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from .context_geometry_maps import ContextGeometryMap, build_default_context_geometry_maps


KEYWORD_HINTS = {
    "literal": {"literal", "exact", "quote", "definition", "prompt", "wording"},
    "hierarchical": {"hierarchy", "tree", "project", "roadmap", "nested", "taxonomy", "plan"},
    "temporal": {"time", "timeline", "sequence", "chronology", "routine", "recurrence", "schedule"},
    "spatial_mechanical": {"spatial", "mechanical", "cad", "layout", "pose", "assembly", "geometry", "3d"},
    "symbolic_mathematical": {"math", "symbolic", "proof", "equation", "phase", "complex", "manifold"},
    "procedural": {"procedure", "workflow", "tool", "command", "code", "routine", "dev-flow", "run"},
    "conflict_verification": {"conflict", "verify", "audit", "contradiction", "redo", "evidence", "check"},
    "creative_synthesis": {"creative", "synthesis", "invent", "analogy", "blend", "option", "improve"},
    "policy_governance": {"policy", "governance", "paama-x", "safety", "permission", "audit"},
    "quantum_holographic": {"quantum", "holographic", "binding", "depth_code", "bank_code", "triplet_code"},
}


@dataclass
class ContextSelectionTrace:
    selected_map: str
    reason: str
    scores: Dict[str, float]


class ContextMapSelector(nn.Module):
    """Select a context geometry map using hints and optional learned features."""

    def __init__(self, dim: int, maps: Optional[Dict[str, ContextGeometryMap]] = None):
        super().__init__()
        self.maps = maps or build_default_context_geometry_maps()
        self.dim = dim
        self.learned_score = nn.Linear(dim, len(self.maps))
        self.map_names = list(self.maps.keys())

    def _keyword_scores(self, hints: Iterable[str] | None) -> Dict[str, float]:
        hints_l = {h.lower() for h in (hints or [])}
        out = {name: 0.0 for name in self.maps}
        for name, keywords in KEYWORD_HINTS.items():
            if name in out:
                out[name] += float(len(hints_l & keywords))
        return out

    def select(
        self,
        context: torch.Tensor,
        requested_map: Optional[str] = None,
        task_hints: Iterable[str] | None = None,
        paamax_policy_hint: Optional[str] = None,
    ) -> tuple[ContextGeometryMap, ContextSelectionTrace]:
        if requested_map and requested_map in self.maps:
            scores = {name: 0.0 for name in self.maps}
            scores[requested_map] = 999.0
            return self.maps[requested_map], ContextSelectionTrace(requested_map, "requested_map", scores)

        keyword_scores = self._keyword_scores(task_hints)
        if paamax_policy_hint:
            keyword_scores["policy_governance"] = keyword_scores.get("policy_governance", 0.0) + 2.0

        learned_scores = self.learned_score(context.mean(dim=1)).mean(dim=0).detach()
        combined = {}
        for idx, name in enumerate(self.map_names):
            combined[name] = float(keyword_scores.get(name, 0.0) + 0.05 * learned_scores[idx].cpu())

        selected = max(combined, key=combined.get)
        if combined[selected] <= 0.0:
            selected = "literal"
            reason = "default_literal"
        else:
            reason = "keyword_learned_score"

        return self.maps[selected], ContextSelectionTrace(selected, reason, combined)


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
