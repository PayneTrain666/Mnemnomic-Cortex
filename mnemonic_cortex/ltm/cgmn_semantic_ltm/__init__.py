"""
Plain-language summary
----------------------
What this file is for: Long-term memory package module:   init  .
How it fits in the system: Supports LTM banks, MANN/geometry helpers, or package wiring used with cortex LTM.
Status: ACTIVE / LEGACY depending on file
Important notes for non-coders: Some files are local copies or aliases; prefer top-level cortex + triple_hybrid for product runtime.

Technical notes (original):
Backward-compatible alias package for CGMN semantic LTM setup.

Re-exports the canonical ``mnemonic_cortex.ltm`` public API so import paths
such as ``mnemonic_cortex.ltm.cgmn_semantic_ltm`` remain stable while using
the shared implementation modules.
"""
from .. import (
    DEFAULT_CGMN_DEPTH_CHART,
    DEFAULT_CURVED_DEPTH_CHART,
    DEFAULT_HG_DEPTH_CHART,
    DEFAULT_MANN_DEPTH_CHART,
    DEFAULT_SPATIAL_DEPTH_CHART,
    ContextToken,
    DepthRouter,
    DualTransformerPolicy,
    EnhancedSpatialMnemonicCortex,
    EpisodeRecord,
    GeometryKeyProjector,
    GeometryMemoryBank,
    HGEpisodicLTM,
    LTMReadResult,
    LTMSubsystem,
    MANNReasoner,
    ReasoningStack,
    SensoryContextBuffer,
    SensoryToken,
    SharedValueStore,
    SharedWriteTrace,
    SpatialLtmMannConfig,
    TransformerBlock,
    TransformerStack,
    TripleHybridLTM,
    WorkingMemory,
    default_config,
    exp_map,
    frechet_mean,
    log_map,
    metric_distance,
    parallel_transport,
)

__all__ = [
    "SpatialLtmMannConfig",
    "default_config",
    "DualTransformerPolicy",
    "DEFAULT_MANN_DEPTH_CHART",
    "DEFAULT_HG_DEPTH_CHART",
    "DEFAULT_CGMN_DEPTH_CHART",
    "DEFAULT_CURVED_DEPTH_CHART",
    "DEFAULT_SPATIAL_DEPTH_CHART",
    "SharedValueStore",
    "SharedWriteTrace",
    "DepthRouter",
    "GeometryKeyProjector",
    "GeometryMemoryBank",
    "LTMReadResult",
    "LTMSubsystem",
    "TripleHybridLTM",
    "EpisodeRecord",
    "HGEpisodicLTM",
    "MANNReasoner",
    "WorkingMemory",
    "SensoryContextBuffer",
    "SensoryToken",
    "ContextToken",
    "EnhancedSpatialMnemonicCortex",
    "TransformerBlock",
    "TransformerStack",
    "ReasoningStack",
    "metric_distance",
    "log_map",
    "exp_map",
    "parallel_transport",
    "frechet_mean",
]
