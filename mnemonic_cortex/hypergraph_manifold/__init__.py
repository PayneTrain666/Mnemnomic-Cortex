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
    "ManifoldRoutingOptions",
    "ManifoldRoutingInput",
    "GeometryDistanceResult",
    "ManifoldRouteAssignment",
    "ManifoldRoutingResult",
    "DepthRetrievalTarget",
    "DepthRetrievalBridgeResult",
    "HGM2ManifoldRoutingResult",
    "compute_geometry_distance",
    "route_hyperedges_to_manifold_charts",
    "build_hgm2_manifold_routing",
    "assign_depth_retrieval_targets",
    "SPCPProceduralOptions",
    "ActionPrimitive",
    "ProceduralActionSequence",
    "ActionSequenceBuildResult",
    "SPCPProcedureEmbedding",
    "SPCPEmbeddingResult",
    "SPCPSimilarityResult",
    "ProceduralMemoryStoreResult",
    "ProceduralMemoryRetrievalCandidate",
    "ProceduralMemoryRetrievalResult",
    "RoboticsPlanningActionOption",
    "RoboticsPlanningBridgeResult",
    "HGM3ProceduralMemoryResult",
    "validate_action_primitive",
    "validate_action_sequence",
    "build_action_sequence_from_hgm2_route",
    "compute_spcp_procedure_embedding",
    "spcp_procedure_similarity",
    "store_procedural_sequences",
    "retrieve_similar_procedures",
    "build_robotics_planning_options",
    "build_hgm3_spcp_procedural_memory",
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

from .hgm2_result import (
    ManifoldRoutingOptions,
    ManifoldRoutingInput,
    GeometryDistanceResult,
    ManifoldRouteAssignment,
    ManifoldRoutingResult,
    DepthRetrievalTarget,
    DepthRetrievalBridgeResult,
    HGM2ManifoldRoutingResult,
)
from .geometry_distance import compute_geometry_distance
from .manifold_router import route_hyperedges_to_manifold_charts, build_hgm2_manifold_routing
from .depth_retrieval import assign_depth_retrieval_targets

from .hgm3_result import (
    SPCPProceduralOptions,
    ActionPrimitive,
    ProceduralActionSequence,
    ActionSequenceBuildResult,
    SPCPProcedureEmbedding,
    SPCPEmbeddingResult,
    SPCPSimilarityResult,
    ProceduralMemoryStoreResult,
    ProceduralMemoryRetrievalCandidate,
    ProceduralMemoryRetrievalResult,
    RoboticsPlanningActionOption,
    RoboticsPlanningBridgeResult,
    HGM3ProceduralMemoryResult,
)
from .action_sequence import validate_action_primitive, validate_action_sequence, build_action_sequence_from_hgm2_route
from .spcp_procedural import compute_spcp_procedure_embedding, spcp_procedure_similarity
from .procedural_memory import store_procedural_sequences, retrieve_similar_procedures
from .robotics_planning_bridge import build_robotics_planning_options, build_hgm3_spcp_procedural_memory

from .hgm4_result import (
    QDTWMBridgeOptions,
    BridgeAdapterStatus,
    HGMBridgePayload,
    BridgePayloadBuildResult,
    SharedSlotLatticeHook,
    SharedSlotLatticeHookResult,
    TraceSafeMemoryPlan,
    BridgeExecutionPreview,
    HGM4BridgeResult,
)
from .qdt_wm_bridge import detect_qdt_wm_adapter_status, build_hgm4_qdt_wm_bridge
from .bridge_payloads import build_bridge_payload_from_hgm_record
from .slot_lattice_hooks import build_shared_slot_lattice_hooks
from .trace_safe_memory_integration import build_trace_safe_memory_plan, preview_bridge_execution

__all__.extend([
    "QDTWMBridgeOptions",
    "BridgeAdapterStatus",
    "HGMBridgePayload",
    "BridgePayloadBuildResult",
    "SharedSlotLatticeHook",
    "SharedSlotLatticeHookResult",
    "TraceSafeMemoryPlan",
    "BridgeExecutionPreview",
    "HGM4BridgeResult",
    "detect_qdt_wm_adapter_status",
    "build_bridge_payload_from_hgm_record",
    "build_shared_slot_lattice_hooks",
    "build_trace_safe_memory_plan",
    "preview_bridge_execution",
    "build_hgm4_qdt_wm_bridge",
])

from .hgm5_result import (
    EmbeddingTrainerOptions,
    HGMEmbeddingRecord,
    EmbeddingTrainerResult,
    BridgeQualityMetric,
    BridgeEvaluationResult,
    IntegrationScore,
    IntegrationScoringResult,
    HGM5EmbeddingEvaluationResult,
)
from .embedding_trainer import build_baseline_hgm_embeddings
from .quality_metrics import evaluate_bridge_payload_quality, evaluate_slot_hook_quality, evaluate_execution_preview_quality
from .integration_scoring import score_hgm_integration_readiness, build_hgm5_embedding_evaluation

__all__.extend([
    "EmbeddingTrainerOptions",
    "HGMEmbeddingRecord",
    "EmbeddingTrainerResult",
    "BridgeQualityMetric",
    "BridgeEvaluationResult",
    "IntegrationScore",
    "IntegrationScoringResult",
    "HGM5EmbeddingEvaluationResult",
    "build_baseline_hgm_embeddings",
    "evaluate_bridge_payload_quality",
    "evaluate_slot_hook_quality",
    "evaluate_execution_preview_quality",
    "score_hgm_integration_readiness",
    "build_hgm5_embedding_evaluation",
])


from .hgm6_result import (
    HGM6CommitOptions,
    WritePermissionState,
    TransactionOperationPreview,
    RollbackOperation,
    RollbackManifest,
    CommitReadinessScore,
    TransactionCommitPreview,
    HGM6WritePermissionResult,
)
from .write_permission_gate import build_write_permission_state, build_hgm6_write_permission_gate
from .transaction_preview import build_transaction_operation_previews, build_transaction_commit_preview
from .rollback_plan import build_rollback_manifest
from .commit_readiness import score_commit_readiness

__all__.extend([
    "HGM6CommitOptions",
    "WritePermissionState",
    "TransactionOperationPreview",
    "RollbackOperation",
    "RollbackManifest",
    "CommitReadinessScore",
    "TransactionCommitPreview",
    "HGM6WritePermissionResult",
    "build_write_permission_state",
    "build_transaction_operation_previews",
    "build_rollback_manifest",
    "score_commit_readiness",
    "build_transaction_commit_preview",
    "build_hgm6_write_permission_gate",
])


from .hgm7_result import (
    HGM7ExecutionOptions,
    WriteExecutionAdapterStatus,
    TransactionLogEntry,
    TransactionLog,
    RecoveryVerificationRecord,
    RecoveryVerificationResult,
    WriteExecutionResult,
    HGM7WriteExecutionResult,
)
from .transaction_log import build_transaction_log
from .recovery_verification import verify_recovery
from .write_execution_adapter import (
    build_write_execution_adapter_status,
    execute_write_adapter,
    build_hgm7_write_execution_adapter,
)

__all__.extend([
    "HGM7ExecutionOptions",
    "WriteExecutionAdapterStatus",
    "TransactionLogEntry",
    "TransactionLog",
    "RecoveryVerificationRecord",
    "RecoveryVerificationResult",
    "WriteExecutionResult",
    "HGM7WriteExecutionResult",
    "build_transaction_log",
    "verify_recovery",
    "build_write_execution_adapter_status",
    "execute_write_adapter",
    "build_hgm7_write_execution_adapter",
])

from .hgm8_result import (
    HGM8RuntimeOptions,
    RuntimeEmbeddingRecord,
    RuntimeEmbeddingTrainerResult,
    SafeWriteReplayRecord,
    SafeWriteReplayResult,
    PipelineBenchmarkMetric,
    PipelineBenchmarkResult,
    HGM8PipelineEvaluationResult,
)
from .runtime_embedding_trainer import build_runtime_embedding_trainer
from .safe_write_replay import evaluate_safe_write_replay
from .pipeline_benchmark import benchmark_hgm_pipeline, build_hgm8_pipeline_evaluation

__all__.extend([
    "HGM8RuntimeOptions",
    "RuntimeEmbeddingRecord",
    "RuntimeEmbeddingTrainerResult",
    "SafeWriteReplayRecord",
    "SafeWriteReplayResult",
    "PipelineBenchmarkMetric",
    "PipelineBenchmarkResult",
    "HGM8PipelineEvaluationResult",
    "build_runtime_embedding_trainer",
    "evaluate_safe_write_replay",
    "benchmark_hgm_pipeline",
    "build_hgm8_pipeline_evaluation",
])

from .hgm9_result import (
    HGM9ReadinessOptions,
    QDTRuntimeReadinessMetric,
    QDTRuntimeEvaluationResult,
    SlotLatticeReplayRecord,
    SlotLatticeReplayBenchmarkResult,
    ProductionReadinessGate,
    HGM9RuntimeIntegrationResult,
)
from .qdt_runtime_evaluation import evaluate_qdt_runtime_integration
from .slot_lattice_replay_benchmark import benchmark_slot_lattice_replay
from .production_readiness_gate import score_production_readiness
from .hgm9_pipeline import build_hgm9_runtime_integration_evaluation

__all__.extend([
    "HGM9ReadinessOptions",
    "QDTRuntimeReadinessMetric",
    "QDTRuntimeEvaluationResult",
    "SlotLatticeReplayRecord",
    "SlotLatticeReplayBenchmarkResult",
    "ProductionReadinessGate",
    "HGM9RuntimeIntegrationResult",
    "evaluate_qdt_runtime_integration",
    "benchmark_slot_lattice_replay",
    "score_production_readiness",
    "build_hgm9_runtime_integration_evaluation",
])

from .hgm10_result import (
    HGM10ReleaseOptions,
    APIFreezeSymbol,
    APIFreezeRecord,
    ReleaseManifestSummary,
    ReleaseConsolidationRecord,
    IntegrationRoadmapItem,
    IntegrationRoadmapRecord,
    HGM10ReleaseConsolidationResult,
)
from .api_freeze import freeze_hgm_public_api
from .release_consolidation import consolidate_hgm_release_manifests
from .integration_roadmap import build_hgm_integration_roadmap
from .hgm10_pipeline import build_hgm10_release_consolidation

__all__.extend([
    "HGM10ReleaseOptions",
    "APIFreezeSymbol",
    "APIFreezeRecord",
    "ReleaseManifestSummary",
    "ReleaseConsolidationRecord",
    "IntegrationRoadmapItem",
    "IntegrationRoadmapRecord",
    "HGM10ReleaseConsolidationResult",
    "freeze_hgm_public_api",
    "consolidate_hgm_release_manifests",
    "build_hgm_integration_roadmap",
    "build_hgm10_release_consolidation",
])
