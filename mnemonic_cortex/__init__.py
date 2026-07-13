"""
Plain-language summary
----------------------
What this file is for: Public package entry that exports parameter-storage loop and CMS depth-stack helpers.
How it fits in the system: This is what `import mnemonic_cortex` exposes first for those advanced storage features.
Status: ACTIVE
Important notes for non-coders: Most of the brain lives in cortex.py and subpackages, not only this file.
"""

# QSPIN PROD-7 generated package marker.

from .parameter_storage_loop_stack import (
    DEFAULT_PARAMETER_LOOP_MANIFOLDS,
    MANIFOLD_STORAGE_FACTORS,
    ParameterStorageLoopConfig,
    ParameterStorageLoopStack,
    estimate_parameter_storage_loop_capacity,
)
from .consolidated_memory_depth_stack import (
    CMS_SUPER_PRODUCT_MANIFOLD,
    CMS_MANIFOLD_STORAGE_FACTORS,
    DEFAULT_CMS_DEPTH_MANIFOLDS,
    DEFAULT_CMS_VISIBLE_MANIFOLD_STACK,
    ConsolidatedMemoryDepthCfg,
    ConsolidatedMemoryDepthStack,
    estimate_consolidated_memory_depth_capacity,
)
from .trainable_parameter_cps import (
    CPSBackedEmbedding,
    CPSBackedLinear,
    CPSBackedMultiheadAttention,
    CommitMetadata,
    ConsolidationCommit,
    ConsolidationEvaluation,
    ConsolidationProposal,
    EvaluationMetadata,
    ParameterCohort,
    ParameterRefMetadata,
    ProposalMetadata,
    RollbackRecord,
    TrainableParameterCPS,
    TrainableParameterCPSConfig,
    TrainableParameterRef,
)

__all__ = [
    "DEFAULT_PARAMETER_LOOP_MANIFOLDS",
    "MANIFOLD_STORAGE_FACTORS",
    "ParameterStorageLoopConfig",
    "ParameterStorageLoopStack",
    "estimate_parameter_storage_loop_capacity",
    "CMS_SUPER_PRODUCT_MANIFOLD",
    "CMS_MANIFOLD_STORAGE_FACTORS",
    "DEFAULT_CMS_DEPTH_MANIFOLDS",
    "DEFAULT_CMS_VISIBLE_MANIFOLD_STACK",
    "ConsolidatedMemoryDepthCfg",
    "ConsolidatedMemoryDepthStack",
    "estimate_consolidated_memory_depth_capacity",
    "CPSBackedEmbedding",
    "CPSBackedLinear",
    "CPSBackedMultiheadAttention",
    "CommitMetadata",
    "ConsolidationCommit",
    "ConsolidationEvaluation",
    "ConsolidationProposal",
    "EvaluationMetadata",
    "ParameterCohort",
    "ParameterRefMetadata",
    "ProposalMetadata",
    "RollbackRecord",
    "TrainableParameterCPS",
    "TrainableParameterCPSConfig",
    "TrainableParameterRef",
]
