"""Configuration schema for HGM/HPME foundation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Tuple

from .enums import GeometryType, ProbabilityNormalizationMode


@dataclass(frozen=True)
class HGMConfig:
    """Runtime-neutral config for HGM/HPME validation and tracing.

    This is not a training config yet. It is a strict schema contract used
    to keep tensors, depth layers, q-spin signatures, and hyperedges sane.
    """

    max_depth_layers: int = 8
    probability_tolerance: float = 1e-5
    qspin_dimension: int = 8
    max_hyperedge_nodes: int = 1024
    allow_singleton_hyperedges: bool = False
    default_normalization: ProbabilityNormalizationMode = ProbabilityNormalizationMode.MUTATION_AXIS
    allowed_geometries: FrozenSet[GeometryType] = field(
        default_factory=lambda: frozenset(GeometryType)
    )
    lineage_tags: Tuple[str, ...] = ("MnemonicCortex", "HGM", "HPME", "HGM-0A")

    def __post_init__(self) -> None:
        if self.max_depth_layers != 8:
            raise ValueError("HGM-0A currently requires exactly 8 depth layers")
        if self.probability_tolerance <= 0:
            raise ValueError("probability_tolerance must be positive")
        if self.qspin_dimension <= 0:
            raise ValueError("qspin_dimension must be positive")
        if self.max_hyperedge_nodes < 2:
            raise ValueError("max_hyperedge_nodes must be >= 2")
