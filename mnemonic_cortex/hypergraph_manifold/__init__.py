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

from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    QDTContractSymbolProbe,
    QDTWriteContractProbeResult,
    TensorProposalPreview,
    ProposalMaterializationContract,
    SlotIDMappingRecord,
    SlotIDMappingPlan,
    QSpinQHConversionRecord,
    QSpinQHConversionContract,
    RollbackSnapshotRequirement,
    RollbackSnapshotHandshake,
    HGMQDTWritePrepResult,
)
from .qdt_write_contract_probe import probe_qdt_wm_write_contracts
from .qdt_proposal_materialization import build_proposal_materialization_contract
from .qdt_slot_mapping_plan import build_slot_id_mapping_plan, sanitize_hgm_slot_id_for_wm, canonical_css_preview
from .qdt_qspin_qh_contract import build_qspin_qh_conversion_contract, depth_to_qdt_index, geometry_to_qdt_map
from .qdt_rollback_handshake import build_rollback_snapshot_handshake
from .hgm_qdt_write_prep_pipeline import build_hgm_qdt_write_prep_contracts

__all__.extend([
    "HGMQDTWritePrepOptions",
    "QDTContractSymbolProbe",
    "QDTWriteContractProbeResult",
    "TensorProposalPreview",
    "ProposalMaterializationContract",
    "SlotIDMappingRecord",
    "SlotIDMappingPlan",
    "QSpinQHConversionRecord",
    "QSpinQHConversionContract",
    "RollbackSnapshotRequirement",
    "RollbackSnapshotHandshake",
    "HGMQDTWritePrepResult",
    "probe_qdt_wm_write_contracts",
    "build_proposal_materialization_contract",
    "build_slot_id_mapping_plan",
    "sanitize_hgm_slot_id_for_wm",
    "canonical_css_preview",
    "build_qspin_qh_conversion_contract",
    "depth_to_qdt_index",
    "geometry_to_qdt_map",
    "build_rollback_snapshot_handshake",
    "build_hgm_qdt_write_prep_contracts",
])

from .hgm_qdt_write_prep2_result import (
    HGMQDTWritePrep2Options,
    DryRunSystemWriteProposalPreview,
    DryRunProposalBuilderResult,
    CommitGatePreflightCheck,
    CommitGatePreflightResult,
    EndToEndWriteSimulationReport,
    HGMQDTWritePrep2Result,
)
from .qdt_dry_run_proposal_builder import build_dry_run_system_write_proposal_previews
from .qdt_commitgate_preflight import run_commitgate_preflight
from .qdt_end_to_end_write_simulation import simulate_end_to_end_hgm_qdt_write, build_hgm_qdt_write_prep_2

__all__.extend([
    "HGMQDTWritePrep2Options",
    "DryRunSystemWriteProposalPreview",
    "DryRunProposalBuilderResult",
    "CommitGatePreflightCheck",
    "CommitGatePreflightResult",
    "EndToEndWriteSimulationReport",
    "HGMQDTWritePrep2Result",
    "build_dry_run_system_write_proposal_previews",
    "run_commitgate_preflight",
    "simulate_end_to_end_hgm_qdt_write",
    "build_hgm_qdt_write_prep_2",
])

from .hgm_qdt_write_prep3_result import (
    HGMQDTWritePrep3Options,
    SyntheticSlotRecord,
    SyntheticSharedSlotStoreSandbox,
    InMemoryCommitGateSimulationOperation,
    InMemoryCommitGateSimulationResult,
    RollbackReplayRecord,
    RollbackReplayVerificationResult,
    HGMQDTWritePrep3Result,
)
from .qdt_synthetic_slot_store import (
    build_synthetic_shared_slot_store_sandbox,
    apply_synthetic_slot_writes,
    restore_synthetic_slot_store_from_previous_state,
)
from .qdt_in_memory_commitgate_simulation import simulate_in_memory_commitgate
from .qdt_rollback_replay_verification import verify_rollback_replay
from .hgm_qdt_write_prep3_pipeline import build_hgm_qdt_write_prep_3

__all__.extend([
    "HGMQDTWritePrep3Options",
    "SyntheticSlotRecord",
    "SyntheticSharedSlotStoreSandbox",
    "InMemoryCommitGateSimulationOperation",
    "InMemoryCommitGateSimulationResult",
    "RollbackReplayRecord",
    "RollbackReplayVerificationResult",
    "HGMQDTWritePrep3Result",
    "build_synthetic_shared_slot_store_sandbox",
    "apply_synthetic_slot_writes",
    "restore_synthetic_slot_store_from_previous_state",
    "simulate_in_memory_commitgate",
    "verify_rollback_replay",
    "build_hgm_qdt_write_prep_3",
])

from .hgm_qdt_write_prep4_result import (
    HGMQDTWritePrep4Options,
    RealContractObjectPreview,
    RealContractObjectConstructionResult,
    SyntheticCommitGateBoundaryCheck,
    SyntheticCommitGateAdapterBoundary,
    RollbackSnapshotBindingRecord,
    RollbackSnapshotBindingPlan,
    HGMQDTWritePrep4Result,
)
from .qdt_real_contract_object_dryrun import build_real_contract_object_previews
from .qdt_synthetic_commitgate_adapter_boundary import build_synthetic_commitgate_adapter_boundary
from .qdt_rollback_snapshot_binding_plan import build_rollback_snapshot_binding_plan
from .hgm_qdt_write_prep4_pipeline import build_hgm_qdt_write_prep_4

__all__.extend([
    "HGMQDTWritePrep4Options",
    "RealContractObjectPreview",
    "RealContractObjectConstructionResult",
    "SyntheticCommitGateBoundaryCheck",
    "SyntheticCommitGateAdapterBoundary",
    "RollbackSnapshotBindingRecord",
    "RollbackSnapshotBindingPlan",
    "HGMQDTWritePrep4Result",
    "build_real_contract_object_previews",
    "build_synthetic_commitgate_adapter_boundary",
    "build_rollback_snapshot_binding_plan",
    "build_hgm_qdt_write_prep_4",
])

from .hgm_qdt_write_prep5_result import (
    HGMQDTWritePrep5Options,
    LiveShapeContractCheck,
    LiveShapeContractHarnessResult,
    PermissionBoundaryAuditCheck,
    PermissionedCommitBoundaryAuditResult,
    ProductionWriteBlocker,
    ProductionWriteBlockerBurnDownResult,
    HGMQDTWritePrep5Result,
)
from .qdt_live_shape_contract_harness import build_live_shape_contract_harness
from .qdt_permission_boundary_audit import audit_permissioned_commit_boundary
from .qdt_production_write_blocker_burndown import build_production_write_blocker_burndown
from .hgm_qdt_write_prep5_pipeline import build_hgm_qdt_write_prep_5

__all__.extend([
    "HGMQDTWritePrep5Options",
    "LiveShapeContractCheck",
    "LiveShapeContractHarnessResult",
    "PermissionBoundaryAuditCheck",
    "PermissionedCommitBoundaryAuditResult",
    "ProductionWriteBlocker",
    "ProductionWriteBlockerBurnDownResult",
    "HGMQDTWritePrep5Result",
    "build_live_shape_contract_harness",
    "audit_permissioned_commit_boundary",
    "build_production_write_blocker_burndown",
    "build_hgm_qdt_write_prep_5",
])

from .hgm_qdt_write_prep6_result import (
    HGMQDTWritePrep6Options,
    SharedSlotStoreParityRecord,
    SharedSlotStoreParityHarnessResult,
    QHStorageRecordSandboxRecord,
    QHStorageRecordSandboxResult,
    RollbackSnapshotBindingDryRunRecord,
    RollbackSnapshotBindingDryRunResult,
    HGMQDTWritePrep6Result,
)
from .qdt_real_shared_slot_store_parity import build_real_shared_slot_store_parity_harness
from .qdt_qh_storage_record_sandbox import build_qh_storage_record_sandbox
from .qdt_rollback_snapshot_binding_dryrun import build_rollback_snapshot_binding_dry_run
from .hgm_qdt_write_prep6_pipeline import build_hgm_qdt_write_prep_6

__all__.extend([
    "HGMQDTWritePrep6Options",
    "SharedSlotStoreParityRecord",
    "SharedSlotStoreParityHarnessResult",
    "QHStorageRecordSandboxRecord",
    "QHStorageRecordSandboxResult",
    "RollbackSnapshotBindingDryRunRecord",
    "RollbackSnapshotBindingDryRunResult",
    "HGMQDTWritePrep6Result",
    "build_real_shared_slot_store_parity_harness",
    "build_qh_storage_record_sandbox",
    "build_rollback_snapshot_binding_dry_run",
    "build_hgm_qdt_write_prep_6",
])

from .hgm_qdt_write_prep7_result import (
    HGMQDTWritePrep7Options,
    PermissionTokenContractRecord,
    PermissionTokenContractResult,
    ShadowCommitSandboxOperation,
    ShadowCommitSandboxResult,
    FinalProductionWriteBlocker,
    FinalProductionWriteReadinessReview,
    HGMQDTWritePrep7Result,
)
from .qdt_permission_token_contract import build_permission_token_contract
from .qdt_shadow_commit_sandbox import build_shadow_commit_sandbox
from .qdt_final_blocker_review import build_final_production_write_readiness_review
from .hgm_qdt_write_prep7_pipeline import build_hgm_qdt_write_prep_7

__all__.extend([
    "HGMQDTWritePrep7Options",
    "PermissionTokenContractRecord",
    "PermissionTokenContractResult",
    "ShadowCommitSandboxOperation",
    "ShadowCommitSandboxResult",
    "FinalProductionWriteBlocker",
    "FinalProductionWriteReadinessReview",
    "HGMQDTWritePrep7Result",
    "build_permission_token_contract",
    "build_shadow_commit_sandbox",
    "build_final_production_write_readiness_review",
    "build_hgm_qdt_write_prep_7",
])
