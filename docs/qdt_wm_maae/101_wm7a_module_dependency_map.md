# WM-7A Module Dependency Map

| module | internal_import_count | external_import_count | internal_modules |
| --- | --- | --- | --- |
| mnemonic_cortex/working_memory/__init__.py | 67 | 0 | .bounded_associative_spread, .context_depth_adapter, .context_geometry_maps, .context_map_selector, .context_stability_guard, .context_to_wm_bridge, .context_trace, .context_triplet_projector, .curvature_metric_policy, .curved_local_trace, .curved_resonant_wm_core, .curved_shadow_write, .curved_slot_state, .depth_specific_addressing, .geometry_aware_addressing, .legacy_enhanced_curved_memory, .qdt_working_memory, .wm_compatibility_wrapper, .wm_config, .wm_conflict_attention, .wm_context_mount, .wm_cortex_integration, .wm_counterfactual_attention, .wm_cross_depth_transformer, .wm_curved_core, .wm_depth_adapters, .wm_depth_fusion, .wm_dual_fusion, .wm_evidence_attention, .wm_external_memory_interfaces, .wm_geometry_linker, .wm_geometry_scoring, .wm_intra_depth_transformer, .wm_ltm_cross_attention, .wm_mann_cross_attention, .wm_memory_augmented_attention, .wm_novelty_attention, .wm_quantum_holographic_storage, .wm_quaternion_depth, .wm_retrieval_lanes, .wm_shadow_write_buffer, .wm_shared_slot_registry, .wm_shared_slot_store, .wm_spcp_cross_attention, .wm_stability, .wm_stability_attention, .wm_system_commit_gate, .wm_trace, .wm_trace_attention, .wm_triplet_state |
| mnemonic_cortex/working_memory/_wm_light_transformer.py | 0 | 4 |  |
| mnemonic_cortex/working_memory/bounded_associative_spread.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/context_depth_adapter.py | 0 | 3 |  |
| mnemonic_cortex/working_memory/context_geometry_maps.py | 0 | 3 |  |
| mnemonic_cortex/working_memory/context_map_selector.py | 1 | 5 | .context_geometry_maps |
| mnemonic_cortex/working_memory/context_stability_guard.py | 0 | 4 |  |
| mnemonic_cortex/working_memory/context_to_wm_bridge.py | 6 | 4 | .context_depth_adapter, .context_geometry_maps, .context_map_selector, .context_stability_guard, .context_trace, .context_triplet_projector |
| mnemonic_cortex/working_memory/context_trace.py | 0 | 3 |  |
| mnemonic_cortex/working_memory/context_triplet_projector.py | 0 | 3 |  |
| mnemonic_cortex/working_memory/curvature_metric_policy.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/curved_local_trace.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/curved_resonant_wm_core.py | 7 | 6 | .bounded_associative_spread, .curvature_metric_policy, .curved_local_trace, .curved_shadow_write, .curved_slot_state, .geometry_aware_addressing, .wm_curved_core |
| mnemonic_cortex/working_memory/curved_shadow_write.py | 2 | 6 | .curved_local_trace, .curved_slot_state |
| mnemonic_cortex/working_memory/curved_slot_state.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/depth_specific_addressing.py | 3 | 6 | .context_geometry_maps, .curvature_metric_policy, .curved_slot_state |
| mnemonic_cortex/working_memory/geometry_aware_addressing.py | 2 | 6 | .curvature_metric_policy, .curved_slot_state |
| mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/qdt_working_memory.py | 18 | 5 | .curvature_metric_policy, .curved_resonant_wm_core, .curved_shadow_write, .curved_slot_state, .depth_specific_addressing, .wm_config, .wm_cross_depth_transformer, .wm_depth_adapters, .wm_depth_fusion, .wm_dual_fusion, .wm_intra_depth_transformer, .wm_memory_augmented_attention, .wm_quantum_holographic_storage, .wm_quaternion_depth, .wm_shared_slot_store, .wm_system_commit_gate, .wm_trace, .wm_triplet_state |
| mnemonic_cortex/working_memory/wm_compatibility_wrapper.py | 2 | 5 | .qdt_working_memory, .wm_config |
| mnemonic_cortex/working_memory/wm_config.py | 0 | 3 |  |
| mnemonic_cortex/working_memory/wm_conflict_attention.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/wm_context_mount.py | 2 | 5 | .context_geometry_maps, .context_to_wm_bridge |
| mnemonic_cortex/working_memory/wm_cortex_integration.py | 3 | 5 | .qdt_working_memory, .wm_compatibility_wrapper, .wm_config |
| mnemonic_cortex/working_memory/wm_counterfactual_attention.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_cross_depth_transformer.py | 1 | 5 | ._wm_light_transformer |
| mnemonic_cortex/working_memory/wm_curved_core.py | 1 | 4 | .legacy_enhanced_curved_memory |
| mnemonic_cortex/working_memory/wm_depth_adapters.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_depth_fusion.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_dual_fusion.py | 3 | 5 | .wm_ltm_cross_attention, .wm_mann_cross_attention, .wm_spcp_cross_attention |
| mnemonic_cortex/working_memory/wm_evidence_attention.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/wm_external_memory_interfaces.py | 1 | 6 | .wm_shared_slot_store |
| mnemonic_cortex/working_memory/wm_geometry_linker.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_geometry_scoring.py | 1 | 6 | .wm_retrieval_lanes |
| mnemonic_cortex/working_memory/wm_intra_depth_transformer.py | 1 | 5 | ._wm_light_transformer |
| mnemonic_cortex/working_memory/wm_ltm_cross_attention.py | 1 | 6 | .wm_external_memory_interfaces |
| mnemonic_cortex/working_memory/wm_mann_cross_attention.py | 1 | 6 | .wm_external_memory_interfaces |
| mnemonic_cortex/working_memory/wm_memory_augmented_attention.py | 10 | 5 | .curved_slot_state, .wm_conflict_attention, .wm_counterfactual_attention, .wm_evidence_attention, .wm_geometry_linker, .wm_geometry_scoring, .wm_novelty_attention, .wm_retrieval_lanes, .wm_stability_attention, .wm_trace_attention |
| mnemonic_cortex/working_memory/wm_novelty_attention.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py | 1 | 8 | .wm_shared_slot_store |
| mnemonic_cortex/working_memory/wm_quaternion_depth.py | 0 | 6 |  |
| mnemonic_cortex/working_memory/wm_retrieval_lanes.py | 1 | 6 | .curved_slot_state |
| mnemonic_cortex/working_memory/wm_shared_slot_registry.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_shared_slot_store.py | 1 | 5 | .wm_shared_slot_registry |
| mnemonic_cortex/working_memory/wm_spcp_cross_attention.py | 1 | 6 | .wm_external_memory_interfaces |
| mnemonic_cortex/working_memory/wm_stability_attention.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_system_commit_gate.py | 3 | 7 | .curved_shadow_write, .wm_quantum_holographic_storage, .wm_shared_slot_store |
| mnemonic_cortex/working_memory/wm_trace.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_trace_attention.py | 0 | 5 |  |
| mnemonic_cortex/working_memory/wm_triplet_state.py | 0 | 5 |  |

## Raw dependency JSON

```json
{
  "mnemonic_cortex/working_memory/__init__.py": {
    "internal_imports": [
      {
        "module": ".legacy_enhanced_curved_memory",
        "names": [
          "EnhancedCurvedMemory"
        ]
      },
      {
        "module": ".wm_curved_core",
        "names": [
          "WMCurvedAssociativeCore"
        ]
      },
      {
        "module": ".curved_resonant_wm_core",
        "names": [
          "CurvedResonanceConfig",
          "CurvedResonantWMCore",
          "CurvedResonanceTrace",
          "ResonanceStepTrace"
        ]
      },
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateConfig",
          "CurvedSlotSnapshot",
          "CurvedSlotStateTrace",
          "CurvedSlotStateBank"
        ]
      },
      {
        "module": ".curvature_metric_policy",
        "names": [
          "CurvatureMetricPolicyConfig",
          "CurvatureMetricPolicyOutput",
          "CurvatureMetricPolicy"
        ]
      },
      {
        "module": ".geometry_aware_addressing",
        "names": [
          "GeometryAwareAddressingConfig",
          "GeometryAwareAddressingTrace",
          "GeometryAwareAddressingOutput",
          "GeometryAwareAddressing"
        ]
      },
      {
        "module": ".bounded_associative_spread",
        "names": [
          "BoundedAssociativeSpreadConfig",
          "BoundedSpreadTrace",
          "BoundedAssociativeSpread"
        ]
      },
      {
        "module": ".curved_local_trace",
        "names": [
          "CurvedTraceEvent",
          "CurvedLocalTrace",
          "CurvedLocalTraceBuilder"
        ]
      },
      {
        "module": ".curved_shadow_write",
        "names": [
          "CurvedShadowWriteConfig",
          "ShadowWriteProposal",
          "ShadowWriteDecision",
          "CurvedShadowWriteBuffer"
        ]
      },
      {
        "module": ".wm_intra_depth_transformer",
        "names": [
          "WMIntraDepthTransformerConfig",
          "WMIntraDepthTransformerTrace",
          "WMIntraDepthTransformer"
        ]
      },
      {
        "module": ".wm_cross_depth_transformer",
        "names": [
          "WMCrossDepthTransformerConfig",
          "WMCrossDepthTransformerTrace",
          "WMCrossDepthTransformer"
        ]
      },
      {
        "module": ".depth_specific_addressing",
        "names": [
          "DepthSpecificAddressingConfig",
          "DepthSpecificAddressingTrace",
          "DepthSpecificAddressingOutput",
          "DepthSpecificAddressing"
        ]
      },
      {
        "module": ".wm_trace",
        "names": [
          "WMTrace",
          "TraceItem",
          "WMTraceEmitter"
        ]
      },
      {
        "module": ".wm_triplet_state",
        "names": [
          "WMTripletStateConfig",
          "WMTripletState",
          "WMTripletProjector"
        ]
      },
      {
        "module": ".wm_depth_adapters",
        "names": [
          "WMDepthAdaptersConfig",
          "WMDepthAdaptersTrace",
          "WMDepthAdapters"
        ]
      },
      {
        "module": ".wm_depth_fusion",
        "names": [
          "WMDepthFusionConfig",
          "WMDepthFusionTrace",
          "WMDepthFusion"
        ]
      },
      {
        "module": ".wm_retrieval_lanes",
        "names": [
          "RetrievalLaneConfig",
          "RetrievalLaneOutput",
          "WMRetrievalLanesOutput",
          "WMRetrievalLanes"
        ]
      },
      {
        "module": ".wm_geometry_scoring",
        "names": [
          "WMGeometryScoringConfig",
          "WMGeometryScoringOutput",
          "WMGeometryScoring"
        ]
      },
      {
        "module": ".wm_geometry_linker",
        "names": [
          "GeometryLink",
          "WMGeometryLinkerConfig",
          "WMGeometryLinker"
        ]
      },
      {
        "module": ".wm_memory_augmented_attention",
        "names": [
          "WMMemoryAugmentedAttentionConfig",
          "WMMemoryAugmentedAttentionOutput",
          "WMMemoryAugmentedAttention"
        ]
      },
      {
        "module": ".wm_evidence_attention",
        "names": [
          "WMEvidenceAttentionConfig",
          "WMEvidenceAttentionOutput",
          "WMEvidenceAttention"
        ]
      },
      {
        "module": ".wm_trace_attention",
        "names": [
          "WMTraceAttentionConfig",
          "WMTraceAttentionOutput",
          "WMTraceAttention"
        ]
      },
      {
        "module": ".wm_counterfactual_attention",
        "names": [
          "WMCounterfactualAttentionConfig",
          "WMCounterfactualAttentionOutput",
          "WMCounterfactualAttention"
        ]
      },
      {
        "module": ".wm_conflict_attention",
        "names": [
          "WMConflictAttentionConfig",
          "WMConflictAttentionOutput",
          "WMConflictAttention"
        ]
      },
      {
        "module": ".wm_novelty_attention",
        "names": [
          "WMNoveltyAttentionConfig",
          "WMNoveltyAttentionOutput",
          "WMNoveltyAttention"
        ]
      },
      {
        "module": ".wm_stability_attention",
        "names": [
          "WMStabilityAttentionConfig",
          "WMStabilityAttentionOutput",
          "WMStabilityAttention"
        ]
      },
      {
        "module": ".wm_external_memory_interfaces",
        "names": [
          "ExternalMemoryQuery",
          "ExternalMemoryResponse",
          "SyntheticExternalMemoryBank"
        ]
      },
      {
        "module": ".wm_ltm_cross_attention",
        "names": [
          "WMLTMCrossAttentionConfig",
          "WMLTMCrossAttentionOutput",
          "WMLTMCrossAttention"
        ]
      },
      {
        "module": ".wm_mann_cross_attention",
        "names": [
          "WMMANNCrossAttentionConfig",
          "WMMANNTraceVisibility",
          "WMMANNCrossAttentionOutput",
          "WMMANNCrossAttention"
        ]
      },
      {
        "module": ".wm_spcp_cross_attention",
        "names": [
          "WMSPCPCrossAttentionConfig",
          "WMSPCPCrossAttentionOutput",
          "WMSPCPCrossAttention"
        ]
      },
      {
        "module": ".wm_dual_fusion",
        "names": [
          "WMDualFusionConfig",
          "WMDualFusionOutput",
          "WMDualFusionController"
        ]
      },
      {
        "module": ".wm_shared_slot_registry",
        "names": [
          "canonical_slot_id",
          "SharedSlotMirrorRef",
          "SharedSlotRecord",
          "SharedSlotRegistry"
        ]
      },
      {
        "module": ".wm_shared_slot_store",
        "names": [
          "tensor_fingerprint",
          "MirroredContentRule",
          "SharedSlotStoreConfig",
          "SharedSlotWriteResult",
          "SharedSlotStore"
        ]
      },
      {
        "module": ".wm_quantum_holographic_storage",
        "names": [
          "GEOMETRY_CODEBOOK",
          "MEMORY_TYPE_CODEBOOK",
          "TASK_MODE_CODEBOOK",
          "QHCodeSchema",
          "build_qh_code_schema",
          "QHInterferenceReport",
          "QHStorageRecord",
          "QuantumHolographicStorageConfig",
          "QuantumHolographicStorage"
        ]
      },
      {
        "module": ".wm_system_commit_gate",
        "names": [
          "SystemWriteProposal",
          "CommitGateDecision",
          "CommitGateEvaluation",
          "SystemCommitGate"
        ]
      },
      {
        "module": ".wm_compatibility_wrapper",
        "names": [
          "QDTWMCompatibilityConfig",
          "QDTWMCompatibilityTrace",
          "QDTWMCompatibilityWrapper"
        ]
      },
      {
        "module": ".wm_cortex_integration",
        "names": [
          "CortexWorkingMemoryIntegrationConfig",
          "CortexWorkingMemoryMigrationResult",
          "EnhancedMnemonicCortexQDTAdapter",
          "build_qdt_working_memory_for_cortex",
          "replace_cortex_working_memory",
          "migration_patch_template"
        ]
      },
      {
        "module": ".wm_config",
        "names": [
          "QDTWorkingMemoryConfig"
        ]
      },
      {
        "module": ".wm_trace",
        "names": [
          "WMTrace",
          "TraceItem"
        ]
      },
      {
        "module": ".wm_triplet_state",
        "names": [
          "WMTripletState",
          "WMTripletProjector"
        ]
      },
      {
        "module": ".wm_quaternion_depth",
        "names": [
          "QuaternionDepthConfig",
          "QuaternionDepthTrace",
          "QuaternionDepthReplicator",
          "normalize_quaternion",
          "quaternion_conjugate",
          "quaternion_multiply",
          "rotate_vectors_by_quaternion"
        ]
      },
      {
        "module": ".wm_intra_depth_transformer",
        "names": [
          "WMIntraDepthTransformer"
        ]
      },
      {
        "module": ".wm_cross_depth_transformer",
        "names": [
          "WMCrossDepthTransformer"
        ]
      },
      {
        "module": ".wm_depth_adapters",
        "names": [
          "WMDepthAdapters"
        ]
      },
      {
        "module": ".wm_depth_fusion",
        "names": [
          "WMDepthFusion"
        ]
      },
      {
        "module": ".context_geometry_maps",
        "names": [
          "ContextGeometryMap",
          "build_default_context_geometry_maps",
          "validate_context_geometry_map"
        ]
      },
      {
        "module": ".context_map_selector",
        "names": [
          "ContextMapSelector",
          "ContextSelectionTrace"
        ]
      },
      {
        "module": ".context_triplet_projector",
        "names": [
          "ContextTripletProjector"
        ]
      },
      {
        "module": ".context_depth_adapter",
        "names": [
          "ContextDepthAdapter"
        ]
      },
      {
        "module": ".context_stability_guard",
        "names": [
          "ContextStabilityGuard",
          "ContextStabilityReport"
        ]
      },
      {
        "module": ".context_trace",
        "names": [
          "ContextMountTrace"
        ]
      },
      {
        "module": ".context_to_wm_bridge",
        "names": [
          "ContextToWMBridge"
        ]
      },
      {
        "module": ".wm_context_mount",
        "names": [
          "GeometryMountedContextBuffer",
          "ContextMapMount"
        ]
      },
      {
        "module": ".wm_geometry_linker",
        "names": [
          "WMGeometryLinker"
        ]
      },
      {
        "module": ".wm_retrieval_lanes",
        "names": [
          "RetrievalLane",
          "RetrievalLaneOutput",
          "WMRetrievalLanes"
        ]
      },
      {
        "module": ".wm_geometry_scoring",
        "names": [
          "WMGeometryScoring"
        ]
      },
      {
        "module": ".wm_memory_augmented_attention",
        "names": [
          "WMMemoryAugmentedAttention"
        ]
      },
      {
        "module": ".wm_evidence_attention",
        "names": [
          "WMEvidenceAttention"
        ]
      },
      {
        "module": ".wm_trace_attention",
        "names": [
          "WMTraceAttention"
        ]
      },
      {
        "module": ".wm_dual_fusion",
        "names": [
          "WMDualFusionController"
        ]
      },
      {
        "module": ".wm_counterfactual_attention",
        "names": [
          "WMCounterfactualAttentionProbe"
        ]
      },
      {
        "module": ".wm_conflict_attention",
        "names": [
          "WMConflictAttention"
        ]
      },
      {
        "module": ".wm_novelty_attention",
        "names": [
          "WMNoveltyAttention"
        ]
      },
      {
        "module": ".wm_stability_attention",
        "names": [
          "WMStabilityAttention"
        ]
      },
      {
        "module": ".wm_shadow_write_buffer",
        "names": [
          "WMShadowWriteBuffer"
        ]
      },
      {
        "module": ".wm_stability",
        "names": [
          "WMStabilityManager"
        ]
      },
      {
        "module": ".qdt_working_memory",
        "names": [
          "QDTWorkingMemory"
        ]
      }
    ],
    "external_imports": []
  },
  "mnemonic_cortex/working_memory/_wm_light_transformer.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "math",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/bounded_associative_spread.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_depth_adapter.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_geometry_maps.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Dict",
          "List",
          "Mapping",
          "Sequence"
        ]
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_map_selector.py": {
    "internal_imports": [
      {
        "module": ".context_geometry_maps",
        "names": [
          "ContextGeometryMap",
          "build_default_context_geometry_maps"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Dict",
          "Iterable",
          "Mapping",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_stability_guard.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Dict",
          "Any"
        ]
      },
      {
        "module": "torch",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_to_wm_bridge.py": {
    "internal_imports": [
      {
        "module": ".context_geometry_maps",
        "names": [
          "ContextGeometryMap",
          "build_default_context_geometry_maps"
        ]
      },
      {
        "module": ".context_map_selector",
        "names": [
          "ContextMapSelector"
        ]
      },
      {
        "module": ".context_triplet_projector",
        "names": [
          "ContextTripletProjector"
        ]
      },
      {
        "module": ".context_depth_adapter",
        "names": [
          "ContextDepthAdapter"
        ]
      },
      {
        "module": ".context_stability_guard",
        "names": [
          "ContextStabilityGuard"
        ]
      },
      {
        "module": ".context_trace",
        "names": [
          "ContextMountTrace"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Iterable",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_trace.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List"
        ]
      }
    ]
  },
  "mnemonic_cortex/working_memory/context_triplet_projector.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/curvature_metric_policy.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/curved_local_trace.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional"
        ]
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/curved_resonant_wm_core.py": {
    "internal_imports": [
      {
        "module": ".wm_curved_core",
        "names": [
          "WMCurvedAssociativeCore"
        ]
      },
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank",
          "CurvedSlotStateConfig"
        ]
      },
      {
        "module": ".curvature_metric_policy",
        "names": [
          "CurvatureMetricPolicy",
          "CurvatureMetricPolicyConfig"
        ]
      },
      {
        "module": ".geometry_aware_addressing",
        "names": [
          "GeometryAwareAddressing"
        ]
      },
      {
        "module": ".bounded_associative_spread",
        "names": [
          "BoundedAssociativeSpread"
        ]
      },
      {
        "module": ".curved_local_trace",
        "names": [
          "CurvedLocalTraceBuilder"
        ]
      },
      {
        "module": ".curved_shadow_write",
        "names": [
          "CurvedShadowWriteBuffer"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/curved_shadow_write.py": {
    "internal_imports": [
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank"
        ]
      },
      {
        "module": ".curved_local_trace",
        "names": [
          "CurvedLocalTrace"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional"
        ]
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/curved_slot_state.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Optional",
          "Sequence"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/depth_specific_addressing.py": {
    "internal_imports": [
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank"
        ]
      },
      {
        "module": ".curvature_metric_policy",
        "names": [
          "CurvatureMetricPolicy",
          "CurvatureMetricPolicyOutput"
        ]
      },
      {
        "module": ".context_geometry_maps",
        "names": [
          "build_default_context_geometry_maps",
          "ContextGeometryMap"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional",
          "Sequence"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/geometry_aware_addressing.py": {
    "internal_imports": [
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank",
          "CurvedSlotSnapshot"
        ]
      },
      {
        "module": ".curvature_metric_policy",
        "names": [
          "CurvatureMetricPolicy",
          "CurvatureMetricPolicyOutput"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Dict",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/qdt_working_memory.py": {
    "internal_imports": [
      {
        "module": ".wm_config",
        "names": [
          "QDTWorkingMemoryConfig"
        ]
      },
      {
        "module": ".wm_trace",
        "names": [
          "WMTraceEmitter"
        ]
      },
      {
        "module": ".wm_triplet_state",
        "names": [
          "WMTripletProjector"
        ]
      },
      {
        "module": ".wm_depth_adapters",
        "names": [
          "WMDepthAdapters",
          "WMDepthAdaptersConfig"
        ]
      },
      {
        "module": ".wm_depth_fusion",
        "names": [
          "WMDepthFusion",
          "WMDepthFusionConfig"
        ]
      },
      {
        "module": ".wm_quaternion_depth",
        "names": [
          "QuaternionDepthReplicator"
        ]
      },
      {
        "module": ".wm_intra_depth_transformer",
        "names": [
          "WMIntraDepthTransformer",
          "WMIntraDepthTransformerConfig"
        ]
      },
      {
        "module": ".wm_cross_depth_transformer",
        "names": [
          "WMCrossDepthTransformer",
          "WMCrossDepthTransformerConfig"
        ]
      },
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank",
          "CurvedSlotStateConfig"
        ]
      },
      {
        "module": ".curvature_metric_policy",
        "names": [
          "CurvatureMetricPolicy",
          "CurvatureMetricPolicyConfig"
        ]
      },
      {
        "module": ".depth_specific_addressing",
        "names": [
          "DepthSpecificAddressing",
          "DepthSpecificAddressingConfig"
        ]
      },
      {
        "module": ".curved_resonant_wm_core",
        "names": [
          "CurvedResonanceConfig",
          "CurvedResonantWMCore"
        ]
      },
      {
        "module": ".curved_shadow_write",
        "names": [
          "CurvedShadowWriteBuffer",
          "CurvedShadowWriteConfig"
        ]
      },
      {
        "module": ".wm_memory_augmented_attention",
        "names": [
          "WMMemoryAugmentedAttention",
          "WMMemoryAugmentedAttentionConfig"
        ]
      },
      {
        "module": ".wm_dual_fusion",
        "names": [
          "WMDualFusionController",
          "WMDualFusionConfig"
        ]
      },
      {
        "module": ".wm_shared_slot_store",
        "names": [
          "SharedSlotStore",
          "SharedSlotStoreConfig"
        ]
      },
      {
        "module": ".wm_quantum_holographic_storage",
        "names": [
          "QuantumHolographicStorage",
          "QuantumHolographicStorageConfig"
        ]
      },
      {
        "module": ".wm_system_commit_gate",
        "names": [
          "SystemCommitGate",
          "SystemWriteProposal"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_compatibility_wrapper.py": {
    "internal_imports": [
      {
        "module": ".wm_config",
        "names": [
          "QDTWorkingMemoryConfig"
        ]
      },
      {
        "module": ".qdt_working_memory",
        "names": [
          "QDTWorkingMemory"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_config.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict"
        ]
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_conflict_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_context_mount.py": {
    "internal_imports": [
      {
        "module": ".context_geometry_maps",
        "names": [
          "ContextGeometryMap",
          "build_default_context_geometry_maps"
        ]
      },
      {
        "module": ".context_to_wm_bridge",
        "names": [
          "ContextToWMBridge"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Dict",
          "List",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_cortex_integration.py": {
    "internal_imports": [
      {
        "module": ".wm_config",
        "names": [
          "QDTWorkingMemoryConfig"
        ]
      },
      {
        "module": ".qdt_working_memory",
        "names": [
          "QDTWorkingMemory"
        ]
      },
      {
        "module": ".wm_compatibility_wrapper",
        "names": [
          "QDTWMCompatibilityConfig",
          "QDTWMCompatibilityWrapper"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_counterfactual_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_cross_depth_transformer.py": {
    "internal_imports": [
      {
        "module": "._wm_light_transformer",
        "names": [
          "LightweightTransformerStack"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_curved_core.py": {
    "internal_imports": [
      {
        "module": ".legacy_enhanced_curved_memory",
        "names": [
          "EnhancedCurvedMemory"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_depth_adapters.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_depth_fusion.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_dual_fusion.py": {
    "internal_imports": [
      {
        "module": ".wm_ltm_cross_attention",
        "names": [
          "WMLTMCrossAttention",
          "WMLTMCrossAttentionConfig"
        ]
      },
      {
        "module": ".wm_mann_cross_attention",
        "names": [
          "WMMANNCrossAttention",
          "WMMANNCrossAttentionConfig"
        ]
      },
      {
        "module": ".wm_spcp_cross_attention",
        "names": [
          "WMSPCPCrossAttention",
          "WMSPCPCrossAttentionConfig"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_evidence_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_external_memory_interfaces.py": {
    "internal_imports": [
      {
        "module": ".wm_shared_slot_store",
        "names": [
          "SharedSlotStore",
          "SharedSlotStoreConfig"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_geometry_linker.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_geometry_scoring.py": {
    "internal_imports": [
      {
        "module": ".wm_retrieval_lanes",
        "names": [
          "WMRetrievalLanesOutput",
          "LANE_NAMES"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_intra_depth_transformer.py": {
    "internal_imports": [
      {
        "module": "._wm_light_transformer",
        "names": [
          "LightweightTransformerStack"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_ltm_cross_attention.py": {
    "internal_imports": [
      {
        "module": ".wm_external_memory_interfaces",
        "names": [
          "ExternalMemoryQuery",
          "ExternalMemoryResponse",
          "SyntheticExternalMemoryBank"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_mann_cross_attention.py": {
    "internal_imports": [
      {
        "module": ".wm_external_memory_interfaces",
        "names": [
          "ExternalMemoryQuery",
          "ExternalMemoryResponse",
          "SyntheticExternalMemoryBank"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_memory_augmented_attention.py": {
    "internal_imports": [
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank"
        ]
      },
      {
        "module": ".wm_retrieval_lanes",
        "names": [
          "RetrievalLaneConfig",
          "WMRetrievalLanes"
        ]
      },
      {
        "module": ".wm_geometry_scoring",
        "names": [
          "WMGeometryScoringConfig",
          "WMGeometryScoring"
        ]
      },
      {
        "module": ".wm_geometry_linker",
        "names": [
          "WMGeometryLinker",
          "WMGeometryLinkerConfig"
        ]
      },
      {
        "module": ".wm_evidence_attention",
        "names": [
          "WMEvidenceAttention",
          "WMEvidenceAttentionConfig"
        ]
      },
      {
        "module": ".wm_trace_attention",
        "names": [
          "WMTraceAttention",
          "WMTraceAttentionConfig"
        ]
      },
      {
        "module": ".wm_counterfactual_attention",
        "names": [
          "WMCounterfactualAttention",
          "WMCounterfactualAttentionConfig"
        ]
      },
      {
        "module": ".wm_conflict_attention",
        "names": [
          "WMConflictAttention",
          "WMConflictAttentionConfig"
        ]
      },
      {
        "module": ".wm_novelty_attention",
        "names": [
          "WMNoveltyAttention",
          "WMNoveltyAttentionConfig"
        ]
      },
      {
        "module": ".wm_stability_attention",
        "names": [
          "WMStabilityAttention",
          "WMStabilityAttentionConfig"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_novelty_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py": {
    "internal_imports": [
      {
        "module": ".wm_shared_slot_store",
        "names": [
          "SharedSlotStore",
          "SharedSlotStoreConfig",
          "tensor_fingerprint"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "hashlib",
        "names": []
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_quaternion_depth.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_retrieval_lanes.py": {
    "internal_imports": [
      {
        "module": ".curved_slot_state",
        "names": [
          "CurvedSlotStateBank"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Mapping",
          "Optional",
          "Sequence",
          "Tuple"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_shared_slot_registry.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Optional",
          "Sequence",
          "Tuple"
        ]
      },
      {
        "module": "hashlib",
        "names": []
      },
      {
        "module": "time",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_shared_slot_store.py": {
    "internal_imports": [
      {
        "module": ".wm_shared_slot_registry",
        "names": [
          "SharedSlotRegistry",
          "SharedSlotRecord",
          "canonical_slot_id"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional",
          "Sequence"
        ]
      },
      {
        "module": "hashlib",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_spcp_cross_attention.py": {
    "internal_imports": [
      {
        "module": ".wm_external_memory_interfaces",
        "names": [
          "ExternalMemoryQuery",
          "ExternalMemoryResponse",
          "SyntheticExternalMemoryBank"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_stability_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_system_commit_gate.py": {
    "internal_imports": [
      {
        "module": ".curved_shadow_write",
        "names": [
          "CurvedShadowWriteBuffer"
        ]
      },
      {
        "module": ".wm_shared_slot_store",
        "names": [
          "SharedSlotStore"
        ]
      },
      {
        "module": ".wm_quantum_holographic_storage",
        "names": [
          "QuantumHolographicStorage",
          "QHStorageRecord"
        ]
      }
    ],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "List",
          "Optional"
        ]
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn.functional",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_trace.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Iterable",
          "List",
          "Optional"
        ]
      },
      {
        "module": "time",
        "names": []
      },
      {
        "module": "uuid",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_trace_attention.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  },
  "mnemonic_cortex/working_memory/wm_triplet_state.py": {
    "internal_imports": [],
    "external_imports": [
      {
        "module": "__future__",
        "names": [
          "annotations"
        ]
      },
      {
        "module": "dataclasses",
        "names": [
          "dataclass",
          "field",
          "asdict"
        ]
      },
      {
        "module": "typing",
        "names": [
          "Any",
          "Dict",
          "Optional"
        ]
      },
      {
        "module": "torch",
        "names": []
      },
      {
        "module": "torch.nn",
        "names": []
      }
    ]
  }
}
```
