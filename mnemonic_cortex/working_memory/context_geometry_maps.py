"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context geometry maps.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_foundation_guards import ensure_finite_tensor, ensure_rank, safe_jsonable, foundation_trace, row_stochastic, clamp_norm

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence


GEOMETRY_SET = {
    "euclidean",
    "hyperbolic",
    "poincare",
    "spherical",
    "torus",
    "complex",
    "complex_projective",
    "cp_kahler",
    "subspace",
    "grassmann",
    "spatial_se3",
    "quaternion",
    "dual_quaternion",
    "spcp",
    "product",
    "fiber_bundle",
    "tangent_bridge",
    "holographic_phase",
}


@dataclass(frozen=True)
class ContextGeometryMap:
    """A context-buffer geometry map.

    The map tells the Geometry-Mounted Context Buffer how to mount context
    information onto QDT-WM depth replicas.

    Depth conventions:
    - default depth count is 8.
    - depth_weights length should match depth_count at runtime, or be padded/truncated.
    - geometry_by_depth describes the primary chart/lens for each depth.
    """

    name: str
    purpose: str
    geometry_by_depth: List[str]
    depth_weights: List[float]
    warmup_weights: List[float]
    trainable_weight_init: List[float]
    reasoning_tags: List[str] = field(default_factory=list)
    paamax_policy_tags: List[str] = field(default_factory=list)
    stability_rules: Dict[str, float] = field(default_factory=dict)
    mount_strategy: str = "additive_bias"
    triplet_bias: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])

    def normalized_for_depths(self, num_depths: int) -> "ContextGeometryMap":
        def fit(values: Sequence[float], fill: float) -> List[float]:
            vals = list(values)
            if len(vals) >= num_depths:
                return vals[:num_depths]
            return vals + [fill] * (num_depths - len(vals))

        def fit_geo(values: Sequence[str]) -> List[str]:
            vals = list(values)
            if len(vals) >= num_depths:
                return vals[:num_depths]
            return vals + ["euclidean"] * (num_depths - len(vals))

        return ContextGeometryMap(
            name=self.name,
            purpose=self.purpose,
            geometry_by_depth=fit_geo(self.geometry_by_depth),
            depth_weights=fit(self.depth_weights, 0.1),
            warmup_weights=fit(self.warmup_weights, 0.1),
            trainable_weight_init=fit(self.trainable_weight_init, 0.1),
            reasoning_tags=list(self.reasoning_tags),
            paamax_policy_tags=list(self.paamax_policy_tags),
            stability_rules=dict(self.stability_rules),
            mount_strategy=self.mount_strategy,
            triplet_bias=fit(self.triplet_bias, 0.25)[:3],
        )


def _base_rules(max_norm: float = 10.0) -> Dict[str, float]:
    return {
        "max_context_norm": max_norm,
        "max_mount_delta": 2.0,
        "max_depth_weight": 1.25,
        "min_depth_weight": 0.0,
        "requires_trace": 1.0,
    }


def build_default_context_geometry_maps(num_depths: int = 8) -> Dict[str, ContextGeometryMap]:
    """Create the full WM-0B context geometry map preset family."""

    raw = {
        "literal": ContextGeometryMap(
            name="literal",
            purpose="Exact wording, current prompt facts, direct context recall.",
            geometry_by_depth=[
                "euclidean", "euclidean", "spherical", "torus",
                "complex", "subspace", "spatial_se3", "spcp",
            ],
            depth_weights=[1.0, 0.6, 0.25, 0.15, 0.15, 0.25, 0.2, 0.1],
            warmup_weights=[1.0, 0.5, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1],
            trainable_weight_init=[0.9, 0.5, 0.2, 0.15, 0.15, 0.25, 0.2, 0.1],
            reasoning_tags=["literal", "exact", "prompt", "quote", "definition"],
            paamax_policy_tags=["low_risk", "source_sensitive"],
            stability_rules=_base_rules(8.0),
        ),
        "hierarchical": ContextGeometryMap(
            name="hierarchical",
            purpose="Nested projects, taxonomies, plans, inheritance, tree-like memory.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "poincare", "spherical",
                "complex_projective", "subspace", "spatial_se3", "spcp",
            ],
            depth_weights=[0.45, 1.0, 0.9, 0.35, 0.25, 0.65, 0.25, 0.2],
            warmup_weights=[0.4, 0.9, 0.75, 0.3, 0.2, 0.55, 0.2, 0.2],
            trainable_weight_init=[0.45, 1.0, 0.85, 0.35, 0.3, 0.65, 0.25, 0.2],
            reasoning_tags=["hierarchy", "tree", "project", "nested", "roadmap", "taxonomy"],
            paamax_policy_tags=["trace_required"],
            stability_rules=_base_rules(10.0),
        ),
        "temporal": ContextGeometryMap(
            name="temporal",
            purpose="Chronology, event order, recurrence, routines, sequences.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex", "subspace", "spatial_se3", "spcp",
            ],
            depth_weights=[0.35, 0.35, 0.85, 1.0, 0.45, 0.55, 0.25, 0.65],
            warmup_weights=[0.3, 0.3, 0.7, 0.9, 0.35, 0.45, 0.2, 0.55],
            trainable_weight_init=[0.35, 0.35, 0.85, 1.0, 0.45, 0.55, 0.25, 0.65],
            reasoning_tags=["time", "sequence", "timeline", "chronology", "routine", "recurrence"],
            paamax_policy_tags=["time_anchor"],
            stability_rules=_base_rules(9.0),
        ),
        "spatial_mechanical": ContextGeometryMap(
            name="spatial_mechanical",
            purpose="Layouts, CAD-like reasoning, mechanisms, pose, physical assembly.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "quaternion", "subspace", "spatial_se3", "dual_quaternion",
            ],
            depth_weights=[0.55, 0.35, 0.45, 0.25, 0.85, 0.7, 1.0, 0.9],
            warmup_weights=[0.45, 0.3, 0.35, 0.2, 0.7, 0.6, 0.9, 0.75],
            trainable_weight_init=[0.55, 0.35, 0.45, 0.25, 0.85, 0.7, 1.0, 0.9],
            reasoning_tags=["spatial", "mechanical", "cad", "layout", "pose", "assembly", "geometry"],
            paamax_policy_tags=["engineering_trace"],
            stability_rules=_base_rules(12.0),
        ),
        "symbolic_mathematical": ContextGeometryMap(
            name="symbolic_mathematical",
            purpose="Mathematics, symbolic structures, phase-coded relations, proofs.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex_projective", "cp_kahler", "subspace", "holographic_phase",
            ],
            depth_weights=[0.45, 0.65, 0.45, 0.55, 1.0, 0.9, 0.85, 0.7],
            warmup_weights=[0.4, 0.55, 0.4, 0.45, 0.85, 0.75, 0.75, 0.6],
            trainable_weight_init=[0.45, 0.65, 0.45, 0.55, 1.0, 0.9, 0.85, 0.7],
            reasoning_tags=["math", "symbolic", "proof", "equation", "phase", "complex"],
            paamax_policy_tags=["precision_required"],
            stability_rules=_base_rules(10.0),
        ),
        "procedural": ContextGeometryMap(
            name="procedural",
            purpose="Commands, workflows, tool-use, code routines, action chains.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex", "subspace", "spatial_se3", "spcp",
            ],
            depth_weights=[0.35, 0.45, 0.45, 0.85, 0.55, 0.8, 0.45, 1.0],
            warmup_weights=[0.3, 0.4, 0.4, 0.75, 0.45, 0.7, 0.4, 0.9],
            trainable_weight_init=[0.35, 0.45, 0.45, 0.85, 0.55, 0.8, 0.45, 1.0],
            reasoning_tags=["procedure", "workflow", "tool", "command", "code", "routine", "dev-flow"],
            paamax_policy_tags=["tool_trace", "write_guard"],
            stability_rules=_base_rules(10.0),
        ),
        "conflict_verification": ContextGeometryMap(
            name="conflict_verification",
            purpose="Contradiction checks, audit, evidence verification, REDO decisions.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "poincare", "spherical",
                "complex", "subspace", "product", "holographic_phase",
            ],
            depth_weights=[0.85, 0.75, 0.7, 0.35, 0.4, 1.0, 0.65, 0.55],
            warmup_weights=[0.75, 0.65, 0.6, 0.3, 0.35, 0.9, 0.55, 0.45],
            trainable_weight_init=[0.85, 0.75, 0.7, 0.35, 0.4, 1.0, 0.65, 0.55],
            reasoning_tags=["conflict", "verify", "audit", "contradiction", "redo", "evidence"],
            paamax_policy_tags=["always_trace", "conflict_gate", "verification_required"],
            stability_rules=_base_rules(8.0),
        ),
        "creative_synthesis": ContextGeometryMap(
            name="creative_synthesis",
            purpose="Invention, analogical transfer, unusual combinations, concept blending.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex_projective", "fiber_bundle", "product", "spcp",
            ],
            depth_weights=[0.35, 0.55, 1.0, 0.85, 0.9, 0.8, 0.75, 0.6],
            warmup_weights=[0.3, 0.45, 0.85, 0.75, 0.8, 0.7, 0.65, 0.5],
            trainable_weight_init=[0.35, 0.55, 1.0, 0.85, 0.9, 0.8, 0.75, 0.6],
            reasoning_tags=["creative", "synthesis", "invent", "analogy", "blend", "option"],
            paamax_policy_tags=["novelty_trace"],
            stability_rules=_base_rules(11.0),
        ),
        "policy_governance": ContextGeometryMap(
            name="policy_governance",
            purpose="PAAMA-X policy routing, safety/trace governance, write permission checks.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex", "subspace", "product", "holographic_phase",
            ],
            depth_weights=[0.9, 0.7, 0.25, 0.25, 0.45, 1.0, 0.6, 0.75],
            warmup_weights=[0.8, 0.6, 0.2, 0.2, 0.4, 0.9, 0.55, 0.65],
            trainable_weight_init=[0.9, 0.7, 0.25, 0.25, 0.45, 1.0, 0.6, 0.75],
            reasoning_tags=["policy", "governance", "paama-x", "safety", "permission", "audit"],
            paamax_policy_tags=["policy_lane", "write_permission", "audit_required"],
            stability_rules=_base_rules(8.0),
        ),
        "quantum_holographic": ContextGeometryMap(
            name="quantum_holographic",
            purpose="Depth/bank/geometry/triplet-coded holographic read/write preparation.",
            geometry_by_depth=[
                "euclidean", "hyperbolic", "spherical", "torus",
                "complex_projective", "holographic_phase", "product", "spcp",
            ],
            depth_weights=[0.4, 0.65, 0.55, 0.65, 0.95, 1.0, 0.8, 0.75],
            warmup_weights=[0.35, 0.55, 0.45, 0.55, 0.85, 0.9, 0.7, 0.65],
            trainable_weight_init=[0.4, 0.65, 0.55, 0.65, 0.95, 1.0, 0.8, 0.75],
            reasoning_tags=["quantum", "holographic", "binding", "code", "depth_code", "memory_code"],
            paamax_policy_tags=["interference_check", "write_guard"],
            stability_rules=_base_rules(9.0),
            mount_strategy="phase_bias",
        ),
    }

    return {name: m.normalized_for_depths(num_depths) for name, m in raw.items()}


def validate_context_geometry_map(map_spec: ContextGeometryMap, num_depths: int = 8) -> None:
    if not map_spec.name:
        raise ValueError("Context map requires a non-empty name")
    if len(map_spec.geometry_by_depth) != num_depths:
        raise ValueError(f"{map_spec.name} geometry_by_depth must have length {num_depths}")
    if len(map_spec.depth_weights) != num_depths:
        raise ValueError(f"{map_spec.name} depth_weights must have length {num_depths}")
    if len(map_spec.warmup_weights) != num_depths:
        raise ValueError(f"{map_spec.name} warmup_weights must have length {num_depths}")
    if len(map_spec.trainable_weight_init) != num_depths:
        raise ValueError(f"{map_spec.name} trainable_weight_init must have length {num_depths}")
    if len(map_spec.triplet_bias) != 3:
        raise ValueError(f"{map_spec.name} triplet_bias must have length 3")
    unknown = [g for g in map_spec.geometry_by_depth if g not in GEOMETRY_SET]
    if unknown:
        raise ValueError(f"{map_spec.name} contains unknown geometries: {unknown}")


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
