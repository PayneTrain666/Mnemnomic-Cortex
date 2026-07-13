"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm geometry linker.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_attention_guards import ensure_attention_query, ensure_candidate_tensor, ensure_attention_scores, stable_softmax, bounded_attention_topk, ensure_lane_output, attention_contract_trace, attention_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


@dataclass
class GeometryLink:
    source_geometry: str
    target_geometry: str
    weight: float = 1.0
    reason: str = "configured"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WMGeometryLinkerConfig:
    dim: int
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")


class WMGeometryLinker(nn.Module):
    """Lightweight geometry-link routing helper.

    WM-3A role:
    - records links between geometry views.
    - supplies lane bias hints to retrieval/attention.
    - does not pretend to perform full manifold transport; that belongs to
      later topology-manager stages.
    """

    def __init__(self, config: WMGeometryLinkerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.router = nn.Linear(config.dim, 7)
        self.links: List[GeometryLink] = [
            GeometryLink("euclidean", "hyperbolic", 0.6, "literal_to_hierarchy"),
            GeometryLink("hyperbolic", "subspace", 0.7, "tree_to_mode"),
            GeometryLink("spatial_se3", "quaternion", 0.8, "spatial_rotation"),
            GeometryLink("spcp", "procedural", 0.9, "procedure_lane"),
            GeometryLink("holographic_phase", "complex_projective", 0.8, "qh_phase"),
        ]
        self.lane_names = ["vector", "hyperbolic", "temporal", "spatial", "procedural", "trace", "policy"]

    def lane_bias(self, query: torch.Tensor) -> Dict[str, float]:
        if query.dim() != 2 or query.size(-1) != self.config.dim:
            raise ValueError(f"Expected query [B,{self.config.dim}], got {tuple(query.shape)}")
        logits = self.router(query).mean(dim=0)
        weights = torch.softmax(logits, dim=0).detach().cpu().tolist()
        return {name: float(weight) for name, weight in zip(self.lane_names, weights)}

    def to_trace(self) -> Dict[str, Any]:
        return {
            "trace_type": "wm_geometry_linker",
            "links": [link.to_dict() for link in self.links],
            "placeholder_notice": "full manifold transport/topology manager is deferred to later topology stages",
        }


# ---------------------------------------------------------------------------
# WM-QD-3A memory-augmented attention quality contract
# ---------------------------------------------------------------------------

def wm_qd3a_attention_contract() -> dict:
    """Return serialization-safe quality metadata for this attention module.

    This no-mutation contract declares candidate schema validation, lane output
    validation, finite geometry-score requirements, bounded attention/top-k
    behavior, trace serialization, PAAMA-X metadata, conflict/quarantine hooks,
    fallback behavior, and compatibility with QDTWorkingMemory.
    """
    return attention_contract_trace(module=__name__)
