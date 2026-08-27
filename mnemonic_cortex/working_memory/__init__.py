"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component:   init  .
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.

Technical notes (original):
QDT-WM-MAAE working-memory package.

Exports are intentionally explicit for preservation tests. Optional modules are
imported independently so one later-stage import issue cannot hide core WM
classes.
"""

from .legacy_enhanced_curved_memory import EnhancedCurvedMemory
from .wm_curved_core import WMCurvedAssociativeCore

# Optional architecture-lock exports.
try:
    from .wm_config import QDTWorkingMemoryCapacityEstimate, QDTWorkingMemoryConfig, qdt_config_from_hardware_profile
except Exception:
    QDTWorkingMemoryConfig = None
    QDTWorkingMemoryCapacityEstimate = None
    qdt_config_from_hardware_profile = None

try:
    from .wm_trace import WMTrace, TraceItem
except Exception:
    WMTrace = None
    TraceItem = None

try:
    from .wm_triplet_state import WMTripletState, WMTripletProjector
except Exception:
    WMTripletState = None
    WMTripletProjector = None

try:
    from .wm_quaternion_depth import QuaternionDepthConfig, QuaternionDepthTrace, QuaternionDepthReplicator, normalize_quaternion, quaternion_conjugate, quaternion_multiply, rotate_vectors_by_quaternion
except Exception:
    QuaternionDepthReplicator = None
    normalize_quaternion = None

try:
    from .wm_intra_depth_transformer import WMIntraDepthTransformer
    from .wm_cross_depth_transformer import WMCrossDepthTransformer
    from .wm_depth_adapters import WMDepthAdapters
    from .wm_depth_fusion import WMDepthFusion
except Exception:
    WMIntraDepthTransformer = None
    WMCrossDepthTransformer = None
    WMDepthAdapters = None
    WMDepthFusion = None

try:
    from .context_geometry_maps import ContextGeometryMap, build_default_context_geometry_maps, validate_context_geometry_map
    from .context_map_selector import ContextMapSelector, ContextSelectionTrace
    from .context_triplet_projector import ContextTripletProjector
    from .context_depth_adapter import ContextDepthAdapter
    from .context_stability_guard import ContextStabilityGuard, ContextStabilityReport
    from .context_trace import ContextMountTrace
    from .context_to_wm_bridge import ContextToWMBridge
    from .wm_context_mount import GeometryMountedContextBuffer, ContextMapMount
except Exception:
    ContextGeometryMap = None
    build_default_context_geometry_maps = None
    validate_context_geometry_map = None
    ContextMapSelector = None
    ContextSelectionTrace = None
    ContextTripletProjector = None
    ContextDepthAdapter = None
    ContextStabilityGuard = None
    ContextStabilityReport = None
    ContextMountTrace = None
    ContextToWMBridge = None
    GeometryMountedContextBuffer = None
    ContextMapMount = None

try:
    from .wm_geometry_linker import WMGeometryLinker
    from .wm_retrieval_lanes import RetrievalLane, RetrievalLaneOutput, WMRetrievalLanes
    from .wm_geometry_scoring import WMGeometryScoring
    from .wm_memory_augmented_attention import WMMemoryAugmentedAttention
    from .wm_evidence_attention import WMEvidenceAttention
    from .wm_trace_attention import WMTraceAttention
    from .wm_dual_fusion import WMDualFusionController
    from .wm_inter_manifold_attention import WMInterManifoldAttention
    from .wm_counterfactual_attention import WMCounterfactualAttentionProbe
    from .wm_conflict_attention import WMConflictAttention
    from .wm_novelty_attention import WMNoveltyAttention
    from .wm_stability_attention import WMStabilityAttention
    from .wm_shadow_write_buffer import WMShadowWriteBuffer
    from .wm_stability import WMStabilityManager
except Exception:
    WMGeometryLinker = None
    RetrievalLane = None
    RetrievalLaneOutput = None
    WMRetrievalLanes = None
    WMGeometryScoring = None
    WMMemoryAugmentedAttention = None
    WMEvidenceAttention = None
    WMTraceAttention = None
    WMDualFusionController = None
    WMInterManifoldAttention = None
    WMCounterfactualAttentionProbe = None
    WMConflictAttention = None
    WMNoveltyAttention = None
    WMStabilityAttention = None
    WMShadowWriteBuffer = None
    WMStabilityManager = None

try:
    from .qdt_working_memory import QDTWorkingMemory
except Exception:
    QDTWorkingMemory = None

try:
    from .qspin_experimental_live_activation import (
        QSpinExperimentalLiveActivationController,
        QSpinExperimentalLiveBlockReason,
        QSpinExperimentalLiveConfig,
        QSpinExperimentalLiveDecision,
        QSpinExperimentalLiveMode,
        QSpinExperimentalLiveRequest,
        QSpinExperimentalLiveResult,
        QSpinExperimentalLiveStatus,
        build_qspin_experimental_live_config,
    )
except Exception:
    QSpinExperimentalLiveActivationController = None
    QSpinExperimentalLiveConfig = None

__all__ = [
    "EnhancedCurvedMemory",
    "WMCurvedAssociativeCore",
    "QDTWorkingMemoryConfig",
    "QDTWorkingMemoryCapacityEstimate",
    "qdt_config_from_hardware_profile",
    "QSpinExperimentalLiveActivationController",
    "QSpinExperimentalLiveBlockReason",
    "QSpinExperimentalLiveConfig",
    "QSpinExperimentalLiveDecision",
    "QSpinExperimentalLiveMode",
    "QSpinExperimentalLiveRequest",
    "QSpinExperimentalLiveResult",
    "QSpinExperimentalLiveStatus",
    "build_qspin_experimental_live_config",
    "WMTrace",
    "TraceItem",
    "WMTripletState",
    "WMTripletProjector",
    "QuaternionDepthReplicator",
    "normalize_quaternion",
    "WMIntraDepthTransformer",
    "WMCrossDepthTransformer",
    "WMDepthAdapters",
    "WMDepthFusion",
    "ContextGeometryMap",
    "build_default_context_geometry_maps",
    "validate_context_geometry_map",
    "ContextMapSelector",
    "ContextSelectionTrace",
    "ContextTripletProjector",
    "ContextDepthAdapter",
    "ContextStabilityGuard",
    "ContextStabilityReport",
    "ContextMountTrace",
    "ContextToWMBridge",
    "GeometryMountedContextBuffer",
    "ContextMapMount",
    "WMGeometryLinker",
    "RetrievalLane",
    "RetrievalLaneOutput",
    "WMRetrievalLanes",
    "WMGeometryScoring",
    "WMMemoryAugmentedAttention",
    "WMEvidenceAttention",
    "WMTraceAttention",
    "WMDualFusionController",
    "WMInterManifoldAttention",
    "WMCounterfactualAttentionProbe",
    "WMConflictAttention",
    "WMNoveltyAttention",
    "WMStabilityAttention",
    "WMShadowWriteBuffer",
    "WMStabilityManager",
    "QDTWorkingMemory",
    "CurvedResonanceConfig",
    "CurvedResonantWMCore",
    "CurvedResonanceTrace",
    "ResonanceStepTrace",
    "CurvedSlotStateConfig",
    "CurvedSlotSnapshot",
    "CurvedSlotStateTrace",
    "CurvedSlotStateBank",
    "CurvatureMetricPolicyConfig",
    "CurvatureMetricPolicyOutput",
    "CurvatureMetricPolicy",
    "GeometryAwareAddressingConfig",
    "GeometryAwareAddressingTrace",
    "GeometryAwareAddressingOutput",
    "GeometryAwareAddressing",
    "BoundedAssociativeSpreadConfig",
    "BoundedSpreadTrace",
    "BoundedAssociativeSpread",
    "CurvedTraceEvent",
    "CurvedLocalTrace",
    "CurvedLocalTraceBuilder",
    "CurvedShadowWriteConfig",
    "ShadowWriteProposal",
    "ShadowWriteDecision",
    "CurvedShadowWriteBuffer",
    "QuaternionDepthConfig",
    "QuaternionDepthTrace",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotate_vectors_by_quaternion",
    "WMIntraDepthTransformerConfig",
    "WMIntraDepthTransformerTrace",
    "WMCrossDepthTransformerConfig",
    "WMCrossDepthTransformerTrace",
    "DepthSpecificAddressingConfig",
    "DepthSpecificAddressingTrace",
    "DepthSpecificAddressingOutput",
    "DepthSpecificAddressing",
    "WMTraceEmitter",
    "WMTripletStateConfig",
    "WMDepthAdaptersConfig",
    "WMDepthAdaptersTrace",
    "WMDepthFusionConfig",
    "WMDepthFusionTrace",
    "RetrievalLaneConfig",
    "WMRetrievalLanesOutput",
    "WMGeometryScoringConfig",
    "WMGeometryScoringOutput",
    "GeometryLink",
    "WMGeometryLinkerConfig",
    "WMMemoryAugmentedAttentionConfig",
    "WMMemoryAugmentedAttentionOutput",
    "WMEvidenceAttentionConfig",
    "WMEvidenceAttentionOutput",
    "WMTraceAttentionConfig",
    "WMTraceAttentionOutput",
    "WMCounterfactualAttentionConfig",
    "WMCounterfactualAttentionOutput",
    "WMCounterfactualAttention",
    "WMConflictAttentionConfig",
    "WMConflictAttentionOutput",
    "WMNoveltyAttentionConfig",
    "WMNoveltyAttentionOutput",
    "WMStabilityAttentionConfig",
    "WMStabilityAttentionOutput",
    "ExternalMemoryQuery",
    "ExternalMemoryResponse",
    "SyntheticExternalMemoryBank",
    "WMLTMCrossAttentionConfig",
    "WMLTMCrossAttentionOutput",
    "WMLTMCrossAttention",
    "WMMANNCrossAttentionConfig",
    "WMMANNTraceVisibility",
    "WMMANNCrossAttentionOutput",
    "WMMANNCrossAttention",
    "WMSPCPCrossAttentionConfig",
    "WMSPCPCrossAttentionOutput",
    "WMSPCPCrossAttention",
    "WMDualFusionConfig",
    "WMDualFusionOutput",
    "WMChartFusionPolicyConfig",
    "WMChartFusionPolicyOutput",
    "WMChartFusionPolicy",
    "scenario_prior",
    "SCENARIO_ROLE_PRIORS",
    "FUSION_SPACE",
    "PreFusionHandoff",
    "prefusion_handoff_contract",
    "WMInterManifoldAttentionConfig",
    "WMInterManifoldAttentionOutput",
    "WMInterManifoldAttention",
    "canonical_slot_id",
    "SharedSlotMirrorRef",
    "SharedSlotRecord",
    "SharedSlotRegistry",
    "tensor_fingerprint",
    "MirroredContentRule",
    "SharedSlotStoreConfig",
    "SharedSlotWriteResult",
    "SharedSlotStore",
    "GEOMETRY_CODEBOOK",
    "MEMORY_TYPE_CODEBOOK",
    "TASK_MODE_CODEBOOK",
    "QHCodeSchema",
    "build_qh_code_schema",
    "QHInterferenceReport",
    "QHStorageRecord",
    "QuantumHolographicStorageConfig",
    "QuantumHolographicStorage",
    "SystemWriteProposal",
    "CommitGateDecision",
    "CommitGateEvaluation",
    "SystemCommitGate",
    "QDTWMCompatibilityConfig",
    "QDTWMCompatibilityTrace",
    "QDTWMCompatibilityWrapper",
    "CortexWorkingMemoryIntegrationConfig",
    "CortexWorkingMemoryMigrationResult",
    "EnhancedMnemonicCortexQDTAdapter",
    "build_qdt_working_memory_for_cortex",
    "replace_cortex_working_memory",
    "migration_patch_template",
]

from .curved_resonant_wm_core import CurvedResonanceConfig, CurvedResonantWMCore, CurvedResonanceTrace, ResonanceStepTrace

from .curved_slot_state import CurvedSlotStateConfig, CurvedSlotSnapshot, CurvedSlotStateTrace, CurvedSlotStateBank
from .curvature_metric_policy import CurvatureMetricPolicyConfig, CurvatureMetricPolicyOutput, CurvatureMetricPolicy

from .geometry_aware_addressing import GeometryAwareAddressingConfig, GeometryAwareAddressingTrace, GeometryAwareAddressingOutput, GeometryAwareAddressing
from .bounded_associative_spread import BoundedAssociativeSpreadConfig, BoundedSpreadTrace, BoundedAssociativeSpread

from .curved_local_trace import CurvedTraceEvent, CurvedLocalTrace, CurvedLocalTraceBuilder
from .curved_shadow_write import CurvedShadowWriteConfig, ShadowWriteProposal, ShadowWriteDecision, CurvedShadowWriteBuffer

from .wm_intra_depth_transformer import WMIntraDepthTransformerConfig, WMIntraDepthTransformerTrace, WMIntraDepthTransformer
from .wm_cross_depth_transformer import WMCrossDepthTransformerConfig, WMCrossDepthTransformerTrace, WMCrossDepthTransformer
from .depth_specific_addressing import DepthSpecificAddressingConfig, DepthSpecificAddressingTrace, DepthSpecificAddressingOutput, DepthSpecificAddressing
from .wm_trace import WMTrace, TraceItem, WMTraceEmitter
from .wm_triplet_state import WMTripletStateConfig, WMTripletState, WMTripletProjector
from .wm_depth_adapters import WMDepthAdaptersConfig, WMDepthAdaptersTrace, WMDepthAdapters
from .wm_depth_fusion import WMDepthFusionConfig, WMDepthFusionTrace, WMDepthFusion
from .wm_retrieval_lanes import RetrievalLaneConfig, RetrievalLaneOutput, WMRetrievalLanesOutput, WMRetrievalLanes
from .wm_geometry_scoring import WMGeometryScoringConfig, WMGeometryScoringOutput, WMGeometryScoring
from .wm_geometry_linker import GeometryLink, WMGeometryLinkerConfig, WMGeometryLinker
from .wm_memory_augmented_attention import WMMemoryAugmentedAttentionConfig, WMMemoryAugmentedAttentionOutput, WMMemoryAugmentedAttention
from .wm_evidence_attention import WMEvidenceAttentionConfig, WMEvidenceAttentionOutput, WMEvidenceAttention
from .wm_trace_attention import WMTraceAttentionConfig, WMTraceAttentionOutput, WMTraceAttention
from .wm_counterfactual_attention import WMCounterfactualAttentionConfig, WMCounterfactualAttentionOutput, WMCounterfactualAttention
from .wm_conflict_attention import WMConflictAttentionConfig, WMConflictAttentionOutput, WMConflictAttention
from .wm_novelty_attention import WMNoveltyAttentionConfig, WMNoveltyAttentionOutput, WMNoveltyAttention
from .wm_stability_attention import WMStabilityAttentionConfig, WMStabilityAttentionOutput, WMStabilityAttention
from .wm_external_memory_interfaces import ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank
from .wm_ltm_cross_attention import WMLTMCrossAttentionConfig, WMLTMCrossAttentionOutput, WMLTMCrossAttention
from .wm_mann_cross_attention import WMMANNCrossAttentionConfig, WMMANNTraceVisibility, WMMANNCrossAttentionOutput, WMMANNCrossAttention
from .wm_spcp_cross_attention import WMSPCPCrossAttentionConfig, WMSPCPCrossAttentionOutput, WMSPCPCrossAttention
from .wm_dual_fusion import WMDualFusionConfig, WMDualFusionOutput, WMDualFusionController
from .wm_chart_fusion_policy import (
    WMChartFusionPolicyConfig,
    WMChartFusionPolicyOutput,
    WMChartFusionPolicy,
    scenario_prior,
    SCENARIO_ROLE_PRIORS,
)
from .wm_prefusion_handoff import (
    FUSION_SPACE,
    PreFusionHandoff,
    prefusion_handoff_contract,
)
from .wm_inter_manifold_attention import (
    WMInterManifoldAttentionConfig,
    WMInterManifoldAttentionOutput,
    WMInterManifoldAttention,
)
from .wm_shared_slot_registry import canonical_slot_id, SharedSlotMirrorRef, SharedSlotRecord, SharedSlotRegistry
from .wm_shared_slot_store import tensor_fingerprint, MirroredContentRule, SharedSlotStoreConfig, SharedSlotWriteResult, SharedSlotStore
from .wm_quantum_holographic_storage import GEOMETRY_CODEBOOK, MEMORY_TYPE_CODEBOOK, TASK_MODE_CODEBOOK, QHCodeSchema, build_qh_code_schema, QHInterferenceReport, QHStorageRecord, QuantumHolographicStorageConfig, QuantumHolographicStorage
from .wm_system_commit_gate import SystemWriteProposal, CommitGateDecision, CommitGateEvaluation, SystemCommitGate
from .wm_compatibility_wrapper import QDTWMCompatibilityConfig, QDTWMCompatibilityTrace, QDTWMCompatibilityWrapper
from .wm_cortex_integration import CortexWorkingMemoryIntegrationConfig, CortexWorkingMemoryMigrationResult, EnhancedMnemonicCortexQDTAdapter, build_qdt_working_memory_for_cortex, replace_cortex_working_memory, migration_patch_template
from .quality import (WMQualitySeverity, WMQualityIssueFamily, WMQualityPatchCategory, WMQualityLineageRef, WMQualityEvidence, WMQualityIssue, WMQualityIssueSet, WMQualityClassifierConfig, WMQualityClassifier, WMQualityRemediationPlanner, WMQualityReport, build_quality_report)

from .wm_foundation_guards import (
    WMFoundationValidationError, ensure_finite_tensor, ensure_rank, ensure_last_dim,
    ensure_shape_prefix, ensure_probability_vector, clamp_norm, safe_jsonable,
    foundation_trace, bounded_topk, row_stochastic,
)

from .wm_depth_guards import (
    TRIPLET_SIZE, WMDepthValidationError, ensure_token_state, ensure_depth_state,
    ensure_triplet_axis, normalize_quaternion, ensure_unit_quaternion,
    ensure_quaternion_pack, depth_summary, token_summary, depth_contract_trace,
    assert_depth_compatible_tokens,
)

from .wm_attention_guards import (
    WMAttentionValidationError, ensure_attention_query, ensure_candidate_tensor,
    ensure_attention_scores, stable_softmax, bounded_attention_topk,
    ensure_lane_output, attention_trace, attention_contract_trace,
    summarize_attention_tensor,
)

from .wm_external_memory_guards import (
    WMExternalMemoryValidationError, ensure_external_memory_response,
    ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id,
    ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record,
    interference_score, external_memory_trace, external_memory_contract_trace,
)

from .wm_commit_cortex_guards import (
    WMCommitCortexValidationError, ensure_commit_proposal_like,
    ensure_commit_decision_like, ensure_rollback_trace,
    ensure_compatibility_input, ensure_migration_template_safety,
    ensure_no_fake_real_source_patch_claim, commit_cortex_trace,
    commit_cortex_contract_trace,
)

# Compatibility alias used by integration tests and older call sites.
# Runtime banks may be synthetic or live triple-hybrid adapters.
try:
    from .wm_triple_hybrid_ltm_adapter import TripleHybridLTMExternalMemoryBank
except Exception:
    TripleHybridLTMExternalMemoryBank = None

if TripleHybridLTMExternalMemoryBank is not None:
    RuntimeExternalMemoryBank = (SyntheticExternalMemoryBank, TripleHybridLTMExternalMemoryBank)
else:
    RuntimeExternalMemoryBank = (SyntheticExternalMemoryBank,)

if "RuntimeExternalMemoryBank" not in __all__:
    __all__.append("RuntimeExternalMemoryBank")

try:
    from .context_compression_memory import (
        ContextCompressionConfig,
        ContextCompressor,
        ContextParameterReferenceExtractor,
        ContextEpisodicMemoryBuilder,
        wm_context_compression_contract,
    )
except Exception:
    ContextCompressionConfig = None
    ContextCompressor = None
    ContextParameterReferenceExtractor = None
    ContextEpisodicMemoryBuilder = None
    wm_context_compression_contract = None

for _name in (
    "ContextCompressionConfig",
    "ContextCompressor",
    "ContextParameterReferenceExtractor",
    "ContextEpisodicMemoryBuilder",
    "wm_context_compression_contract",
):
    if _name not in __all__:
        __all__.append(_name)
