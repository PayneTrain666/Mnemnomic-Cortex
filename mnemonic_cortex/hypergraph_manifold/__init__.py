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

from .normalization import normalize_probability_payload
from .probability_expander import (
    build_probability_expansion,
    expand_from_mutation_tokens,
    infer_probability_shape_contract,
    validate_probability_payload,
)
from .runtime_result import (
    NormalizationReport,
    ProbabilityExpansionResult,
    ScenarioCandidate,
    ScenarioExtractionResult,
)
from .scenario_extraction import extract_top_k_scenarios

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
    "normalize_probability_payload",
    "build_probability_expansion",
    "expand_from_mutation_tokens",
    "infer_probability_shape_contract",
    "validate_probability_payload",
    "NormalizationReport",
    "ProbabilityExpansionResult",
    "ScenarioCandidate",
    "ScenarioExtractionResult",
    "extract_top_k_scenarios",
    "HyperedgeBindingOptions",
    "HyperedgeBindingInput",
    "BoundScenarioHyperedge",
    "HyperedgeBindingResult",
    "CoherenceScoreReport",
    "ConflictEdge",
    "ConflictGraphResult",
    "OpportunityEdge",
    "OpportunityGraphResult",
    "HGM1ScenarioGraphResult",
    "score_hyperedge_coherence",
    "bind_scenario_candidates",
    "build_hgm1_scenario_graph",
    "detect_conflict_edges",
    "detect_opportunity_edges",
]

from .hgm1_result import (
    HyperedgeBindingOptions,
    HyperedgeBindingInput,
    BoundScenarioHyperedge,
    HyperedgeBindingResult,
    CoherenceScoreReport,
    ConflictEdge,
    ConflictGraphResult,
    OpportunityEdge,
    OpportunityGraphResult,
    HGM1ScenarioGraphResult,
)
from .coherence import score_hyperedge_coherence
from .hyperedge_binder import bind_scenario_candidates, build_hgm1_scenario_graph
from .conflict_graph import detect_conflict_edges
from .opportunity_graph import detect_opportunity_edges
