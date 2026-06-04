"""HGM/HPME foundation package.

HGM-0A adds typed, validation-first primitives for Hypergraph Manifold
spaces and Hyperset Probability Matrix Expansion. It is additive and does
not mutate existing Mnemonic Cortex working-memory code paths.
"""

from .config import HGMConfig
from .enums import (
    DepthLayer,
    GeometryType,
    HyperedgeKind,
    MutationDirection,
    ProbabilityNormalizationMode,
    TraceEventKind,
    ValidationSeverity,
)
from .shapes import (
    P_VM,
    P_VMD,
    P_VMDC,
    P_VMDCT,
    P_VMDCTA,
    SHAPE_CONTRACTS,
    TensorShapeContract,
    contract_by_name,
)
from .types import (
    DepthLayerAssignment,
    HypersetMatrix,
    MagnitudeBin,
    ManifoldChart,
    MutationToken,
    QSpinSignature,
    ScenarioHyperedge,
    TraceRecord,
)
from .validation import ValidationMessage, ValidationResult

__all__ = [
    "HGMConfig",
    "DepthLayer",
    "GeometryType",
    "HyperedgeKind",
    "MutationDirection",
    "ProbabilityNormalizationMode",
    "TraceEventKind",
    "ValidationSeverity",
    "P_VM",
    "P_VMD",
    "P_VMDC",
    "P_VMDCT",
    "P_VMDCTA",
    "SHAPE_CONTRACTS",
    "TensorShapeContract",
    "contract_by_name",
    "DepthLayerAssignment",
    "HypersetMatrix",
    "MagnitudeBin",
    "ManifoldChart",
    "MutationToken",
    "QSpinSignature",
    "ScenarioHyperedge",
    "TraceRecord",
    "ValidationMessage",
    "ValidationResult",
]
