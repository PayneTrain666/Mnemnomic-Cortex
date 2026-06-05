# WM-QD-6A API Surface Audit

## Token budget / scope

```json
{
  "stage": "WM-QD-6A",
  "target": "Final quality-deepened release/readiness pack, audit, benchmarks, production readiness, deferred hardening plan, and campaign closure.",
  "minimum_complete_version": [
    "API/source audit",
    "module dependency audit",
    "full pytest rerun",
    "smoke benchmark rerun",
    "contract verification across WM-QD-1A through WM-QD-5A",
    "release manifest",
    "production readiness document",
    "deferred hardening plan",
    "quality tracker/deferred updates",
    "final ZIP package"
  ],
  "deep_implementation_version": [
    "AST-based public API inventory",
    "AST-based import/dependency inventory",
    "contract invocation audit for all WM-QD contracts",
    "runtime benchmark harness for QDT read/process/write, wrapper, shared/QH growth",
    "full source/test/doc/release manifest",
    "honest production caveats",
    "remaining real-source integration requirements",
    "final campaign closure gate"
  ],
  "estimated_source_module_count": 55,
  "estimated_test_count": 51,
  "expected_doc_count": "10+ final WM-QD-6A docs/outputs",
  "benchmark_count": 4,
  "split_decision": "No sub-split required; generated full artifacts in files and summarized in final response.",
  "explicit_out_of_scope": [
    "Patching real EnhancedMnemonicCortex source because the real source file is not present",
    "Persistent database/backend implementation",
    "Real external LTM/MANN/SPCP adapter implementation",
    "Real quantum/holographic hardware backend",
    "Distributed/concurrent transaction semantics",
    "Long-running hardware benchmark suite"
  ]
}
```

## Public API by module

| module | classes | functions | dataclasses |
| --- | --- | --- | --- |
| mnemonic_cortex/working_memory/__init__.py |  |  |  |
| mnemonic_cortex/working_memory/_wm_light_transformer.py | LightweightTransformerBlock, LightweightTransformerStack |  |  |
| mnemonic_cortex/working_memory/bounded_associative_spread.py | BoundedAssociativeSpreadConfig, BoundedSpreadTrace, BoundedAssociativeSpread | wm_qd1a_foundation_contract | BoundedAssociativeSpreadConfig, BoundedSpreadTrace |
| mnemonic_cortex/working_memory/context_depth_adapter.py | ContextDepthAdapter |  |  |
| mnemonic_cortex/working_memory/context_geometry_maps.py | ContextGeometryMap | build_default_context_geometry_maps, validate_context_geometry_map, wm_qd1a_foundation_contract | ContextGeometryMap |
| mnemonic_cortex/working_memory/context_map_selector.py | ContextSelectionTrace, ContextMapSelector | wm_qd1a_foundation_contract | ContextSelectionTrace |
| mnemonic_cortex/working_memory/context_stability_guard.py | ContextStabilityReport, ContextStabilityGuard |  | ContextStabilityReport |
| mnemonic_cortex/working_memory/context_to_wm_bridge.py | ContextToWMBridge | wm_qd1a_foundation_contract |  |
| mnemonic_cortex/working_memory/context_trace.py | ContextMountTrace |  | ContextMountTrace |
| mnemonic_cortex/working_memory/context_triplet_projector.py | ContextTripletProjector |  |  |
| mnemonic_cortex/working_memory/curvature_metric_policy.py | CurvatureMetricPolicyConfig, CurvatureMetricPolicyOutput, CurvatureMetricPolicy | wm_qd1a_foundation_contract | CurvatureMetricPolicyConfig, CurvatureMetricPolicyOutput |
| mnemonic_cortex/working_memory/curved_local_trace.py | CurvedTraceEvent, CurvedLocalTrace, CurvedLocalTraceBuilder | wm_qd1a_foundation_contract | CurvedTraceEvent, CurvedLocalTrace |
| mnemonic_cortex/working_memory/curved_resonant_wm_core.py | ResonanceStepTrace, CurvedResonanceTrace, CurvedResonanceConfig, CurvedResonantWMCore | wm_qd1a_foundation_contract | ResonanceStepTrace, CurvedResonanceTrace, CurvedResonanceConfig |
| mnemonic_cortex/working_memory/curved_shadow_write.py | CurvedShadowWriteConfig, ShadowWriteProposal, ShadowWriteDecision, CurvedShadowWriteBuffer | wm_qd1a_foundation_contract | CurvedShadowWriteConfig, ShadowWriteProposal, ShadowWriteDecision |
| mnemonic_cortex/working_memory/curved_slot_state.py | CurvedSlotStateConfig, CurvedSlotSnapshot, CurvedSlotStateTrace, CurvedSlotStateBank | wm_qd1a_foundation_contract | CurvedSlotStateConfig, CurvedSlotSnapshot, CurvedSlotStateTrace |
| mnemonic_cortex/working_memory/depth_specific_addressing.py | DepthSpecificAddressingConfig, DepthSpecificAddressingTrace, DepthSpecificAddressingOutput, DepthSpecificAddressing | wm_qd2a_depth_contract | DepthSpecificAddressingConfig, DepthSpecificAddressingTrace, DepthSpecificAddressingOutput |
| mnemonic_cortex/working_memory/geometry_aware_addressing.py | GeometryAwareAddressingConfig, GeometryAwareAddressingTrace, GeometryAwareAddressingOutput, GeometryAwareAddressing | wm_qd1a_foundation_contract | GeometryAwareAddressingConfig, GeometryAwareAddressingTrace, GeometryAwareAddressingOutput |
| mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py | CurvedMemoryReadTrace, EnhancedCurvedMemory | wm_qd1a_foundation_contract | CurvedMemoryReadTrace |
| mnemonic_cortex/working_memory/qdt_working_memory.py | QDTWorkingMemory | wm_qd2a_depth_contract, wm_qd5a_commit_cortex_contract |  |
| mnemonic_cortex/working_memory/wm_attention_guards.py | WMAttentionValidationError | ensure_attention_query, ensure_candidate_tensor, ensure_attention_scores, stable_softmax, bounded_attention_topk, ensure_lane_output, attention_trace, attention_contract_trace, summarize_attention_tensor |  |
| mnemonic_cortex/working_memory/wm_commit_cortex_guards.py | WMCommitCortexValidationError | ensure_commit_proposal_like, ensure_commit_decision_like, ensure_rollback_trace, ensure_compatibility_input, ensure_migration_template_safety, ensure_no_fake_real_source_patch_claim, commit_cortex_trace, commit_cortex_contract_trace |  |
| mnemonic_cortex/working_memory/wm_compatibility_wrapper.py | QDTWMCompatibilityConfig, QDTWMCompatibilityTrace, QDTWMCompatibilityWrapper | wm_qd5a_commit_cortex_contract | QDTWMCompatibilityConfig, QDTWMCompatibilityTrace |
| mnemonic_cortex/working_memory/wm_config.py | QDTWorkingMemoryConfig |  | QDTWorkingMemoryConfig |
| mnemonic_cortex/working_memory/wm_conflict_attention.py | WMConflictAttentionConfig, WMConflictAttentionOutput, WMConflictAttention | wm_qd3a_attention_contract | WMConflictAttentionConfig, WMConflictAttentionOutput |
| mnemonic_cortex/working_memory/wm_context_mount.py | GeometryMountedContextBuffer | wm_qd1a_foundation_contract |  |
| mnemonic_cortex/working_memory/wm_cortex_integration.py | CortexWorkingMemoryIntegrationConfig, CortexWorkingMemoryMigrationResult, EnhancedMnemonicCortexQDTAdapter | build_qdt_working_memory_for_cortex, replace_cortex_working_memory, migration_patch_template, wm_qd5a_commit_cortex_contract | CortexWorkingMemoryIntegrationConfig, CortexWorkingMemoryMigrationResult |
| mnemonic_cortex/working_memory/wm_counterfactual_attention.py | WMCounterfactualAttentionConfig, WMCounterfactualAttentionOutput, WMCounterfactualAttention | wm_qd3a_attention_contract | WMCounterfactualAttentionConfig, WMCounterfactualAttentionOutput |
| mnemonic_cortex/working_memory/wm_cross_depth_transformer.py | WMCrossDepthTransformerConfig, WMCrossDepthTransformerTrace, WMCrossDepthTransformer | wm_qd2a_depth_contract | WMCrossDepthTransformerConfig, WMCrossDepthTransformerTrace |
| mnemonic_cortex/working_memory/wm_curved_core.py | WMCurvedAssociativeCore | wm_qd1a_foundation_contract |  |
| mnemonic_cortex/working_memory/wm_depth_adapters.py | WMDepthAdaptersConfig, WMDepthAdaptersTrace, WMDepthAdapters | wm_qd2a_depth_contract | WMDepthAdaptersConfig, WMDepthAdaptersTrace |
| mnemonic_cortex/working_memory/wm_depth_fusion.py | WMDepthFusionConfig, WMDepthFusionTrace, WMDepthFusion | wm_qd2a_depth_contract | WMDepthFusionConfig, WMDepthFusionTrace |
| mnemonic_cortex/working_memory/wm_depth_guards.py | WMDepthValidationError | ensure_token_state, ensure_depth_state, ensure_triplet_axis, normalize_quaternion, ensure_unit_quaternion, ensure_quaternion_pack, depth_summary, token_summary, depth_contract_trace, assert_depth_compatible_tokens |  |
| mnemonic_cortex/working_memory/wm_dual_fusion.py | WMDualFusionConfig, WMDualFusionOutput, WMDualFusionController | wm_qd4a_external_memory_contract | WMDualFusionConfig, WMDualFusionOutput |
| mnemonic_cortex/working_memory/wm_evidence_attention.py | WMEvidenceAttentionConfig, WMEvidenceAttentionOutput, WMEvidenceAttention | wm_qd3a_attention_contract | WMEvidenceAttentionConfig, WMEvidenceAttentionOutput |
| mnemonic_cortex/working_memory/wm_external_memory_guards.py | WMExternalMemoryValidationError | ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_trace, external_memory_contract_trace |  |
| mnemonic_cortex/working_memory/wm_external_memory_interfaces.py | ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank | wm_qd4a_external_memory_contract | ExternalMemoryQuery, ExternalMemoryResponse |
| mnemonic_cortex/working_memory/wm_foundation_guards.py | WMFoundationValidationError | ensure_finite_tensor, ensure_rank, ensure_last_dim, ensure_shape_prefix, ensure_probability_vector, clamp_norm, safe_jsonable, foundation_trace, bounded_topk, row_stochastic |  |
| mnemonic_cortex/working_memory/wm_geometry_linker.py | GeometryLink, WMGeometryLinkerConfig, WMGeometryLinker | wm_qd3a_attention_contract | GeometryLink, WMGeometryLinkerConfig |
| mnemonic_cortex/working_memory/wm_geometry_scoring.py | WMGeometryScoringConfig, WMGeometryScoringOutput, WMGeometryScoring | wm_qd3a_attention_contract | WMGeometryScoringConfig, WMGeometryScoringOutput |
| mnemonic_cortex/working_memory/wm_intra_depth_transformer.py | WMIntraDepthTransformerConfig, WMIntraDepthTransformerTrace, WMIntraDepthTransformer | wm_qd2a_depth_contract | WMIntraDepthTransformerConfig, WMIntraDepthTransformerTrace |
| mnemonic_cortex/working_memory/wm_ltm_cross_attention.py | WMLTMCrossAttentionConfig, WMLTMCrossAttentionOutput, WMLTMCrossAttention | wm_qd4a_external_memory_contract | WMLTMCrossAttentionConfig, WMLTMCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_mann_cross_attention.py | WMMANNCrossAttentionConfig, WMMANNTraceVisibility, WMMANNCrossAttentionOutput, WMMANNCrossAttention | wm_qd4a_external_memory_contract | WMMANNCrossAttentionConfig, WMMANNTraceVisibility, WMMANNCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_memory_augmented_attention.py | WMMemoryAugmentedAttentionConfig, WMMemoryAugmentedAttentionOutput, WMMemoryAugmentedAttention | wm_qd3a_attention_contract | WMMemoryAugmentedAttentionConfig, WMMemoryAugmentedAttentionOutput |
| mnemonic_cortex/working_memory/wm_novelty_attention.py | WMNoveltyAttentionConfig, WMNoveltyAttentionOutput, WMNoveltyAttention | wm_qd3a_attention_contract | WMNoveltyAttentionConfig, WMNoveltyAttentionOutput |
| mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py | QHCodeSchema, QHInterferenceReport, QHStorageRecord, QuantumHolographicStorageConfig, QuantumHolographicStorage | build_qh_code_schema, wm_qd4a_external_memory_contract | QHCodeSchema, QHInterferenceReport, QHStorageRecord, QuantumHolographicStorageConfig |
| mnemonic_cortex/working_memory/wm_quaternion_depth.py | QuaternionDepthTrace, QuaternionDepthConfig, QuaternionDepthReplicator | normalize_quaternion, quaternion_conjugate, quaternion_multiply, rotate_vectors_by_quaternion, wm_qd2a_depth_contract | QuaternionDepthTrace, QuaternionDepthConfig |
| mnemonic_cortex/working_memory/wm_retrieval_lanes.py | RetrievalLaneConfig, RetrievalLaneOutput, WMRetrievalLanesOutput, WMRetrievalLanes | wm_qd3a_attention_contract | RetrievalLaneConfig, RetrievalLaneOutput, WMRetrievalLanesOutput |
| mnemonic_cortex/working_memory/wm_shared_slot_registry.py | SharedSlotMirrorRef, SharedSlotRecord, SharedSlotRegistry | canonical_slot_id, wm_qd4a_external_memory_contract | SharedSlotMirrorRef, SharedSlotRecord |
| mnemonic_cortex/working_memory/wm_shared_slot_store.py | MirroredContentRule, SharedSlotStoreConfig, SharedSlotWriteResult, SharedSlotStore | tensor_fingerprint, wm_qd4a_external_memory_contract | MirroredContentRule, SharedSlotStoreConfig, SharedSlotWriteResult |
| mnemonic_cortex/working_memory/wm_spcp_cross_attention.py | WMSPCPCrossAttentionConfig, WMSPCPCrossAttentionOutput, WMSPCPCrossAttention | wm_qd4a_external_memory_contract | WMSPCPCrossAttentionConfig, WMSPCPCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_stability_attention.py | WMStabilityAttentionConfig, WMStabilityAttentionOutput, WMStabilityAttention | wm_qd3a_attention_contract | WMStabilityAttentionConfig, WMStabilityAttentionOutput |
| mnemonic_cortex/working_memory/wm_system_commit_gate.py | SystemWriteProposal, CommitGateDecision, CommitGateEvaluation, SystemCommitGate | wm_qd5a_commit_cortex_contract | SystemWriteProposal, CommitGateDecision, CommitGateEvaluation |
| mnemonic_cortex/working_memory/wm_trace.py | TraceItem, WMTrace, WMTraceEmitter | wm_qd2a_depth_contract | TraceItem, WMTrace |
| mnemonic_cortex/working_memory/wm_trace_attention.py | WMTraceAttentionConfig, WMTraceAttentionOutput, WMTraceAttention | wm_qd3a_attention_contract | WMTraceAttentionConfig, WMTraceAttentionOutput |
| mnemonic_cortex/working_memory/wm_triplet_state.py | WMTripletStateConfig, WMTripletState, WMTripletProjector | wm_qd2a_depth_contract | WMTripletStateConfig, WMTripletState |

## Raw API audit

```json
{
  "mnemonic_cortex/working_memory/__init__.py": {
    "classes": [],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/_wm_light_transformer.py": {
    "classes": [
      {
        "name": "LightweightTransformerBlock",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 8
      },
      {
        "name": "LightweightTransformerStack",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 51
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/bounded_associative_spread.py": {
    "classes": [
      {
        "name": "BoundedAssociativeSpreadConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "BoundedSpreadTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 40
      },
      {
        "name": "BoundedAssociativeSpread",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "transition_matrix",
          "forward",
          "hebbian_update",
          "validate_transition"
        ],
        "lineno": 54
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 186
      }
    ],
    "dataclasses": [
      "BoundedAssociativeSpreadConfig",
      "BoundedSpreadTrace"
    ]
  },
  "mnemonic_cortex/working_memory/context_depth_adapter.py": {
    "classes": [
      {
        "name": "ContextDepthAdapter",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 7
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/context_geometry_maps.py": {
    "classes": [
      {
        "name": "ContextGeometryMap",
        "bases": [],
        "decorators": [
          "dataclass(frozen=True)"
        ],
        "methods": [
          "normalized_for_depths"
        ],
        "lineno": 32
      }
    ],
    "functions": [
      {
        "name": "build_default_context_geometry_maps",
        "args": [
          "num_depths"
        ],
        "lineno": 94
      },
      {
        "name": "validate_context_geometry_map",
        "args": [
          "map_spec",
          "num_depths"
        ],
        "lineno": 244
      },
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 266
      }
    ],
    "dataclasses": [
      "ContextGeometryMap"
    ]
  },
  "mnemonic_cortex/working_memory/context_map_selector.py": {
    "classes": [
      {
        "name": "ContextSelectionTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [],
        "lineno": 29
      },
      {
        "name": "ContextMapSelector",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "select"
        ],
        "lineno": 35
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 88
      }
    ],
    "dataclasses": [
      "ContextSelectionTrace"
    ]
  },
  "mnemonic_cortex/working_memory/context_stability_guard.py": {
    "classes": [
      {
        "name": "ContextStabilityReport",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 10
      },
      {
        "name": "ContextStabilityGuard",
        "bases": [],
        "decorators": [],
        "methods": [
          "check",
          "repair"
        ],
        "lineno": 29
      }
    ],
    "functions": [],
    "dataclasses": [
      "ContextStabilityReport"
    ]
  },
  "mnemonic_cortex/working_memory/context_to_wm_bridge.py": {
    "classes": [
      {
        "name": "ContextToWMBridge",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 18
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 82
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/context_trace.py": {
    "classes": [
      {
        "name": "ContextMountTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 8
      }
    ],
    "functions": [],
    "dataclasses": [
      "ContextMountTrace"
    ]
  },
  "mnemonic_cortex/working_memory/context_triplet_projector.py": {
    "classes": [
      {
        "name": "ContextTripletProjector",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "project",
          "fuse",
          "forward"
        ],
        "lineno": 7
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/curvature_metric_policy.py": {
    "classes": [
      {
        "name": "CurvatureMetricPolicyConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 14
      },
      {
        "name": "CurvatureMetricPolicyOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 39
      },
      {
        "name": "CurvatureMetricPolicy",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "clamped_components",
          "drift_penalty",
          "set_reference_to_current",
          "context_conditioned_curvature",
          "forward",
          "repair_in_place",
          "validate_policy"
        ],
        "lineno": 60
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 199
      }
    ],
    "dataclasses": [
      "CurvatureMetricPolicyConfig",
      "CurvatureMetricPolicyOutput"
    ]
  },
  "mnemonic_cortex/working_memory/curved_local_trace.py": {
    "classes": [
      {
        "name": "CurvedTraceEvent",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 12
      },
      {
        "name": "CurvedLocalTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "add_event",
          "set_write_decision",
          "merge_paamax",
          "to_dict"
        ],
        "lineno": 25
      },
      {
        "name": "CurvedLocalTraceBuilder",
        "bases": [],
        "decorators": [],
        "methods": [
          "from_resonance_trace",
          "with_geometry_map",
          "with_curvature_state",
          "with_depth_contribution",
          "with_disagreement",
          "build"
        ],
        "lineno": 74
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 132
      }
    ],
    "dataclasses": [
      "CurvedTraceEvent",
      "CurvedLocalTrace"
    ]
  },
  "mnemonic_cortex/working_memory/curved_resonant_wm_core.py": {
    "classes": [
      {
        "name": "ResonanceStepTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 22
      },
      {
        "name": "CurvedResonanceTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 34
      },
      {
        "name": "CurvedResonanceConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 53
      },
      {
        "name": "CurvedResonantWMCore",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 81
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 371
      }
    ],
    "dataclasses": [
      "ResonanceStepTrace",
      "CurvedResonanceTrace",
      "CurvedResonanceConfig"
    ]
  },
  "mnemonic_cortex/working_memory/curved_shadow_write.py": {
    "classes": [
      {
        "name": "CurvedShadowWriteConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 17
      },
      {
        "name": "ShadowWriteProposal",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 39
      },
      {
        "name": "ShadowWriteDecision",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 75
      },
      {
        "name": "CurvedShadowWriteBuffer",
        "bases": [],
        "decorators": [],
        "methods": [
          "stage",
          "interference_score",
          "evaluate",
          "commit",
          "reject",
          "pending_count",
          "to_dict"
        ],
        "lineno": 88
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 277
      }
    ],
    "dataclasses": [
      "CurvedShadowWriteConfig",
      "ShadowWriteProposal",
      "ShadowWriteDecision"
    ]
  },
  "mnemonic_cortex/working_memory/curved_slot_state.py": {
    "classes": [
      {
        "name": "CurvedSlotStateConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 14
      },
      {
        "name": "CurvedSlotSnapshot",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "shape_summary"
        ],
        "lineno": 45
      },
      {
        "name": "CurvedSlotStateTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 73
      },
      {
        "name": "CurvedSlotStateBank",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "stable_tensors",
          "repair_in_place",
          "validate_state",
          "snapshot",
          "update_slots",
          "forward"
        ],
        "lineno": 84
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 264
      }
    ],
    "dataclasses": [
      "CurvedSlotStateConfig",
      "CurvedSlotSnapshot",
      "CurvedSlotStateTrace"
    ]
  },
  "mnemonic_cortex/working_memory/depth_specific_addressing.py": {
    "classes": [
      {
        "name": "DepthSpecificAddressingConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 18
      },
      {
        "name": "DepthSpecificAddressingTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 55
      },
      {
        "name": "DepthSpecificAddressingOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 71
      },
      {
        "name": "DepthSpecificAddressing",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 112
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 282
      }
    ],
    "dataclasses": [
      "DepthSpecificAddressingConfig",
      "DepthSpecificAddressingTrace",
      "DepthSpecificAddressingOutput"
    ]
  },
  "mnemonic_cortex/working_memory/geometry_aware_addressing.py": {
    "classes": [
      {
        "name": "GeometryAwareAddressingConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 17
      },
      {
        "name": "GeometryAwareAddressingTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 59
      },
      {
        "name": "GeometryAwareAddressingOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 72
      },
      {
        "name": "GeometryAwareAddressing",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 91
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 272
      }
    ],
    "dataclasses": [
      "GeometryAwareAddressingConfig",
      "GeometryAwareAddressingTrace",
      "GeometryAwareAddressingOutput"
    ]
  },
  "mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py": {
    "classes": [
      {
        "name": "CurvedMemoryReadTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 14
      },
      {
        "name": "EnhancedCurvedMemory",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "enable_energy_efficient_mode",
          "forward"
        ],
        "lineno": 31
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 214
      }
    ],
    "dataclasses": [
      "CurvedMemoryReadTrace"
    ]
  },
  "mnemonic_cortex/working_memory/qdt_working_memory.py": {
    "classes": [
      {
        "name": "QDTWorkingMemory",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 33
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 312
      },
      {
        "name": "wm_qd5a_commit_cortex_contract",
        "args": [],
        "lineno": 327
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_attention_guards.py": {
    "classes": [
      {
        "name": "WMAttentionValidationError",
        "bases": [
          "WMFoundationValidationError"
        ],
        "decorators": [],
        "methods": [],
        "lineno": 19
      }
    ],
    "functions": [
      {
        "name": "ensure_attention_query",
        "args": [
          "name",
          "tensor",
          "expected_dim"
        ],
        "lineno": 27
      },
      {
        "name": "ensure_candidate_tensor",
        "args": [
          "name",
          "tensor"
        ],
        "lineno": 42
      },
      {
        "name": "ensure_attention_scores",
        "args": [
          "name",
          "scores"
        ],
        "lineno": 64
      },
      {
        "name": "stable_softmax",
        "args": [
          "scores",
          "dim",
          "temperature"
        ],
        "lineno": 83
      },
      {
        "name": "bounded_attention_topk",
        "args": [
          "scores",
          "k",
          "dim"
        ],
        "lineno": 95
      },
      {
        "name": "ensure_lane_output",
        "args": [
          "lane_name",
          "output"
        ],
        "lineno": 104
      },
      {
        "name": "attention_trace",
        "args": [],
        "lineno": 133
      },
      {
        "name": "attention_contract_trace",
        "args": [],
        "lineno": 169
      },
      {
        "name": "summarize_attention_tensor",
        "args": [
          "name",
          "tensor"
        ],
        "lineno": 191
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_commit_cortex_guards.py": {
    "classes": [
      {
        "name": "WMCommitCortexValidationError",
        "bases": [
          "WMFoundationValidationError"
        ],
        "decorators": [],
        "methods": [],
        "lineno": 17
      }
    ],
    "functions": [
      {
        "name": "ensure_commit_proposal_like",
        "args": [
          "name",
          "proposal",
          "expected_dim"
        ],
        "lineno": 33
      },
      {
        "name": "ensure_commit_decision_like",
        "args": [
          "name",
          "decision"
        ],
        "lineno": 70
      },
      {
        "name": "ensure_rollback_trace",
        "args": [
          "name",
          "trace"
        ],
        "lineno": 92
      },
      {
        "name": "ensure_compatibility_input",
        "args": [
          "name",
          "tensor",
          "expected_dim"
        ],
        "lineno": 105
      },
      {
        "name": "ensure_migration_template_safety",
        "args": [
          "name",
          "template"
        ],
        "lineno": 114
      },
      {
        "name": "ensure_no_fake_real_source_patch_claim",
        "args": [
          "name",
          "payload"
        ],
        "lineno": 129
      },
      {
        "name": "commit_cortex_trace",
        "args": [],
        "lineno": 140
      },
      {
        "name": "commit_cortex_contract_trace",
        "args": [],
        "lineno": 176
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_compatibility_wrapper.py": {
    "classes": [
      {
        "name": "QDTWMCompatibilityConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 16
      },
      {
        "name": "QDTWMCompatibilityTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 44
      },
      {
        "name": "QDTWMCompatibilityWrapper",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "input_dim",
          "hidden_dim",
          "route",
          "forward",
          "read",
          "process",
          "write",
          "stability_report"
        ],
        "lineno": 56
      }
    ],
    "functions": [
      {
        "name": "wm_qd5a_commit_cortex_contract",
        "args": [],
        "lineno": 170
      }
    ],
    "dataclasses": [
      "QDTWMCompatibilityConfig",
      "QDTWMCompatibilityTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_config.py": {
    "classes": [
      {
        "name": "QDTWorkingMemoryConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_dict"
        ],
        "lineno": 8
      }
    ],
    "functions": [],
    "dataclasses": [
      "QDTWorkingMemoryConfig"
    ]
  },
  "mnemonic_cortex/working_memory/wm_conflict_attention.py": {
    "classes": [
      {
        "name": "WMConflictAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 14
      },
      {
        "name": "WMConflictAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 30
      },
      {
        "name": "WMConflictAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 45
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 95
      }
    ],
    "dataclasses": [
      "WMConflictAttentionConfig",
      "WMConflictAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_context_mount.py": {
    "classes": [
      {
        "name": "GeometryMountedContextBuffer",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "select_map",
          "mount"
        ],
        "lineno": 19
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 75
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_cortex_integration.py": {
    "classes": [
      {
        "name": "CortexWorkingMemoryIntegrationConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "qdt_config",
          "compatibility_config"
        ],
        "lineno": 17
      },
      {
        "name": "CortexWorkingMemoryMigrationResult",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 66
      },
      {
        "name": "EnhancedMnemonicCortexQDTAdapter",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "read_working_memory",
          "process_working_memory",
          "write_working_memory",
          "forward"
        ],
        "lineno": 78
      }
    ],
    "functions": [
      {
        "name": "build_qdt_working_memory_for_cortex",
        "args": [
          "config"
        ],
        "lineno": 117
      },
      {
        "name": "replace_cortex_working_memory",
        "args": [
          "cortex",
          "config"
        ],
        "lineno": 124
      },
      {
        "name": "migration_patch_template",
        "args": [
          "config"
        ],
        "lineno": 163
      },
      {
        "name": "wm_qd5a_commit_cortex_contract",
        "args": [],
        "lineno": 192
      }
    ],
    "dataclasses": [
      "CortexWorkingMemoryIntegrationConfig",
      "CortexWorkingMemoryMigrationResult"
    ]
  },
  "mnemonic_cortex/working_memory/wm_counterfactual_attention.py": {
    "classes": [
      {
        "name": "WMCounterfactualAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMCounterfactualAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 26
      },
      {
        "name": "WMCounterfactualAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 41
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 95
      }
    ],
    "dataclasses": [
      "WMCounterfactualAttentionConfig",
      "WMCounterfactualAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_cross_depth_transformer.py": {
    "classes": [
      {
        "name": "WMCrossDepthTransformerConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 15
      },
      {
        "name": "WMCrossDepthTransformerTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 55
      },
      {
        "name": "WMCrossDepthTransformer",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 70
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 155
      }
    ],
    "dataclasses": [
      "WMCrossDepthTransformerConfig",
      "WMCrossDepthTransformerTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_curved_core.py": {
    "classes": [
      {
        "name": "WMCurvedAssociativeCore",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "last_trace",
          "enable_energy_efficient_mode",
          "forward"
        ],
        "lineno": 13
      }
    ],
    "functions": [
      {
        "name": "wm_qd1a_foundation_contract",
        "args": [],
        "lineno": 99
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_depth_adapters.py": {
    "classes": [
      {
        "name": "WMDepthAdaptersConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMDepthAdaptersTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 33
      },
      {
        "name": "WMDepthAdapters",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 44
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 110
      }
    ],
    "dataclasses": [
      "WMDepthAdaptersConfig",
      "WMDepthAdaptersTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_depth_fusion.py": {
    "classes": [
      {
        "name": "WMDepthFusionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMDepthFusionTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 32
      },
      {
        "name": "WMDepthFusion",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 45
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 109
      }
    ],
    "dataclasses": [
      "WMDepthFusionConfig",
      "WMDepthFusionTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_depth_guards.py": {
    "classes": [
      {
        "name": "WMDepthValidationError",
        "bases": [
          "WMFoundationValidationError"
        ],
        "decorators": [],
        "methods": [],
        "lineno": 21
      }
    ],
    "functions": [
      {
        "name": "ensure_token_state",
        "args": [
          "name",
          "tensor",
          "expected_dim"
        ],
        "lineno": 25
      },
      {
        "name": "ensure_depth_state",
        "args": [
          "name",
          "tensor"
        ],
        "lineno": 36
      },
      {
        "name": "ensure_triplet_axis",
        "args": [
          "name",
          "tensor",
          "triplet_axis",
          "triplet_size"
        ],
        "lineno": 65
      },
      {
        "name": "normalize_quaternion",
        "args": [
          "quaternion",
          "eps"
        ],
        "lineno": 73
      },
      {
        "name": "ensure_unit_quaternion",
        "args": [
          "name",
          "quaternion",
          "atol"
        ],
        "lineno": 81
      },
      {
        "name": "ensure_quaternion_pack",
        "args": [
          "name",
          "quaternion"
        ],
        "lineno": 97
      },
      {
        "name": "depth_summary",
        "args": [
          "depth_state"
        ],
        "lineno": 111
      },
      {
        "name": "token_summary",
        "args": [
          "token_state"
        ],
        "lineno": 127
      },
      {
        "name": "depth_contract_trace",
        "args": [],
        "lineno": 141
      },
      {
        "name": "assert_depth_compatible_tokens",
        "args": [
          "depth_state",
          "token_state"
        ],
        "lineno": 178
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_dual_fusion.py": {
    "classes": [
      {
        "name": "WMDualFusionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 17
      },
      {
        "name": "WMDualFusionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 50
      },
      {
        "name": "WMDualFusionController",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 69
      }
    ],
    "functions": [
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 185
      }
    ],
    "dataclasses": [
      "WMDualFusionConfig",
      "WMDualFusionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_evidence_attention.py": {
    "classes": [
      {
        "name": "WMEvidenceAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 14
      },
      {
        "name": "WMEvidenceAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 38
      },
      {
        "name": "WMEvidenceAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 53
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 125
      }
    ],
    "dataclasses": [
      "WMEvidenceAttentionConfig",
      "WMEvidenceAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_external_memory_guards.py": {
    "classes": [
      {
        "name": "WMExternalMemoryValidationError",
        "bases": [
          "WMFoundationValidationError"
        ],
        "decorators": [],
        "methods": [],
        "lineno": 19
      }
    ],
    "functions": [
      {
        "name": "ensure_external_memory_response",
        "args": [
          "name",
          "response"
        ],
        "lineno": 27
      },
      {
        "name": "ensure_mann_trace_visibility",
        "args": [
          "name",
          "trace"
        ],
        "lineno": 60
      },
      {
        "name": "ensure_fusion_inputs",
        "args": [
          "ltm",
          "mann",
          "spcp"
        ],
        "lineno": 75
      },
      {
        "name": "ensure_shared_slot_id",
        "args": [
          "name",
          "slot_id"
        ],
        "lineno": 94
      },
      {
        "name": "ensure_shared_slot_record",
        "args": [
          "name",
          "record"
        ],
        "lineno": 101
      },
      {
        "name": "ensure_qh_code_schema",
        "args": [
          "name",
          "schema"
        ],
        "lineno": 114
      },
      {
        "name": "ensure_qh_storage_record",
        "args": [
          "name",
          "record"
        ],
        "lineno": 129
      },
      {
        "name": "interference_score",
        "args": [
          "a",
          "b",
          "eps"
        ],
        "lineno": 146
      },
      {
        "name": "external_memory_trace",
        "args": [],
        "lineno": 160
      },
      {
        "name": "external_memory_contract_trace",
        "args": [],
        "lineno": 199
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_external_memory_interfaces.py": {
    "classes": [
      {
        "name": "ExternalMemoryQuery",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_trace"
        ],
        "lineno": 19
      },
      {
        "name": "ExternalMemoryResponse",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_dict"
        ],
        "lineno": 69
      },
      {
        "name": "SyntheticExternalMemoryBank",
        "bases": [],
        "decorators": [],
        "methods": [
          "query"
        ],
        "lineno": 136
      }
    ],
    "functions": [
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 232
      }
    ],
    "dataclasses": [
      "ExternalMemoryQuery",
      "ExternalMemoryResponse"
    ]
  },
  "mnemonic_cortex/working_memory/wm_foundation_guards.py": {
    "classes": [
      {
        "name": "WMFoundationValidationError",
        "bases": [
          "ValueError"
        ],
        "decorators": [],
        "methods": [],
        "lineno": 10
      }
    ],
    "functions": [
      {
        "name": "ensure_finite_tensor",
        "args": [
          "name",
          "tensor"
        ],
        "lineno": 14
      },
      {
        "name": "ensure_rank",
        "args": [
          "name",
          "tensor",
          "rank"
        ],
        "lineno": 23
      },
      {
        "name": "ensure_last_dim",
        "args": [
          "name",
          "tensor",
          "dim"
        ],
        "lineno": 30
      },
      {
        "name": "ensure_shape_prefix",
        "args": [
          "name",
          "tensor",
          "prefix"
        ],
        "lineno": 37
      },
      {
        "name": "ensure_probability_vector",
        "args": [
          "name",
          "tensor",
          "dim",
          "atol"
        ],
        "lineno": 46
      },
      {
        "name": "clamp_norm",
        "args": [
          "tensor",
          "max_norm",
          "eps"
        ],
        "lineno": 56
      },
      {
        "name": "safe_jsonable",
        "args": [
          "value"
        ],
        "lineno": 63
      },
      {
        "name": "foundation_trace",
        "args": [],
        "lineno": 88
      },
      {
        "name": "bounded_topk",
        "args": [
          "scores",
          "k",
          "dim"
        ],
        "lineno": 113
      },
      {
        "name": "row_stochastic",
        "args": [
          "matrix",
          "eps"
        ],
        "lineno": 121
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_geometry_linker.py": {
    "classes": [
      {
        "name": "GeometryLink",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 13
      },
      {
        "name": "WMGeometryLinkerConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 24
      },
      {
        "name": "WMGeometryLinker",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "lane_bias",
          "to_trace"
        ],
        "lineno": 33
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 76
      }
    ],
    "dataclasses": [
      "GeometryLink",
      "WMGeometryLinkerConfig"
    ]
  },
  "mnemonic_cortex/working_memory/wm_geometry_scoring.py": {
    "classes": [
      {
        "name": "WMGeometryScoringConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 16
      },
      {
        "name": "WMGeometryScoringOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 33
      },
      {
        "name": "WMGeometryScoring",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 52
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 140
      }
    ],
    "dataclasses": [
      "WMGeometryScoringConfig",
      "WMGeometryScoringOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_intra_depth_transformer.py": {
    "classes": [
      {
        "name": "WMIntraDepthTransformerConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 15
      },
      {
        "name": "WMIntraDepthTransformerTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 55
      },
      {
        "name": "WMIntraDepthTransformer",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 69
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 154
      }
    ],
    "dataclasses": [
      "WMIntraDepthTransformerConfig",
      "WMIntraDepthTransformerTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_ltm_cross_attention.py": {
    "classes": [
      {
        "name": "WMLTMCrossAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 16
      },
      {
        "name": "WMLTMCrossAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 32
      },
      {
        "name": "WMLTMCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 47
      }
    ],
    "functions": [
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 100
      }
    ],
    "dataclasses": [
      "WMLTMCrossAttentionConfig",
      "WMLTMCrossAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_mann_cross_attention.py": {
    "classes": [
      {
        "name": "WMMANNCrossAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 16
      },
      {
        "name": "WMMANNTraceVisibility",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 35
      },
      {
        "name": "WMMANNCrossAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 53
      },
      {
        "name": "WMMANNCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 70
      }
    ],
    "functions": [
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 133
      }
    ],
    "dataclasses": [
      "WMMANNCrossAttentionConfig",
      "WMMANNTraceVisibility",
      "WMMANNCrossAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_memory_augmented_attention.py": {
    "classes": [
      {
        "name": "WMMemoryAugmentedAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 24
      },
      {
        "name": "WMMemoryAugmentedAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 42
      },
      {
        "name": "WMMemoryAugmentedAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 63
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 204
      }
    ],
    "dataclasses": [
      "WMMemoryAugmentedAttentionConfig",
      "WMMemoryAugmentedAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_novelty_attention.py": {
    "classes": [
      {
        "name": "WMNoveltyAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 14
      },
      {
        "name": "WMNoveltyAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 30
      },
      {
        "name": "WMNoveltyAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 45
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 92
      }
    ],
    "dataclasses": [
      "WMNoveltyAttentionConfig",
      "WMNoveltyAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py": {
    "classes": [
      {
        "name": "QHCodeSchema",
        "bases": [],
        "decorators": [
          "dataclass(frozen=True)"
        ],
        "methods": [
          "to_tuple",
          "composite_code",
          "to_dict"
        ],
        "lineno": 67
      },
      {
        "name": "QHInterferenceReport",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 140
      },
      {
        "name": "QHStorageRecord",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_dict"
        ],
        "lineno": 155
      },
      {
        "name": "QuantumHolographicStorageConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 205
      },
      {
        "name": "QuantumHolographicStorage",
        "bases": [],
        "decorators": [],
        "methods": [
          "check_interference",
          "create_record",
          "create_from_shared_slot",
          "trace_summary",
          "to_dict"
        ],
        "lineno": 223
      }
    ],
    "functions": [
      {
        "name": "build_qh_code_schema",
        "args": [],
        "lineno": 109
      },
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 392
      }
    ],
    "dataclasses": [
      "QHCodeSchema",
      "QHInterferenceReport",
      "QHStorageRecord",
      "QuantumHolographicStorageConfig"
    ]
  },
  "mnemonic_cortex/working_memory/wm_quaternion_depth.py": {
    "classes": [
      {
        "name": "QuaternionDepthTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 72
      },
      {
        "name": "QuaternionDepthConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 90
      },
      {
        "name": "QuaternionDepthReplicator",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "normalized_depth_quaternions",
          "replicate",
          "forward",
          "depth_consistency_report",
          "set_depth_quaternion",
          "dual_quaternion_status"
        ],
        "lineno": 109
      }
    ],
    "functions": [
      {
        "name": "normalize_quaternion",
        "args": [
          "q",
          "eps"
        ],
        "lineno": 13
      },
      {
        "name": "quaternion_conjugate",
        "args": [
          "q"
        ],
        "lineno": 27
      },
      {
        "name": "quaternion_multiply",
        "args": [
          "a",
          "b"
        ],
        "lineno": 35
      },
      {
        "name": "rotate_vectors_by_quaternion",
        "args": [
          "v",
          "q",
          "eps"
        ],
        "lineno": 52
      },
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 283
      }
    ],
    "dataclasses": [
      "QuaternionDepthTrace",
      "QuaternionDepthConfig"
    ]
  },
  "mnemonic_cortex/working_memory/wm_retrieval_lanes.py": {
    "classes": [
      {
        "name": "RetrievalLaneConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 27
      },
      {
        "name": "RetrievalLaneOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 59
      },
      {
        "name": "WMRetrievalLanesOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 77
      },
      {
        "name": "WMRetrievalLanes",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward",
          "stability_report"
        ],
        "lineno": 90
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 241
      }
    ],
    "dataclasses": [
      "RetrievalLaneConfig",
      "RetrievalLaneOutput",
      "WMRetrievalLanesOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_shared_slot_registry.py": {
    "classes": [
      {
        "name": "SharedSlotMirrorRef",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_dict"
        ],
        "lineno": 31
      },
      {
        "name": "SharedSlotRecord",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "add_or_update_mirror",
          "set_conflict",
          "grant_write_permission",
          "to_dict"
        ],
        "lineno": 55
      },
      {
        "name": "SharedSlotRegistry",
        "bases": [],
        "decorators": [],
        "methods": [
          "get_or_create",
          "link_mirror",
          "mark_conflict",
          "grant_write",
          "get",
          "by_memory_type",
          "to_dict",
          "trace_summary"
        ],
        "lineno": 120
      }
    ],
    "functions": [
      {
        "name": "canonical_slot_id",
        "args": [
          "namespace",
          "local_slot_id",
          "content_fingerprint"
        ],
        "lineno": 15
      },
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 243
      }
    ],
    "dataclasses": [
      "SharedSlotMirrorRef",
      "SharedSlotRecord"
    ]
  },
  "mnemonic_cortex/working_memory/wm_shared_slot_store.py": {
    "classes": [
      {
        "name": "MirroredContentRule",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "attach_qh_record",
          "qh_trace_for_slot",
          "to_dict"
        ],
        "lineno": 24
      },
      {
        "name": "SharedSlotStoreConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 83
      },
      {
        "name": "SharedSlotWriteResult",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "attach_qh_record",
          "qh_trace_for_slot",
          "to_dict"
        ],
        "lineno": 97
      },
      {
        "name": "SharedSlotStore",
        "bases": [],
        "decorators": [],
        "methods": [
          "validate_rules",
          "write_slot",
          "mirror_slot",
          "mark_conflict",
          "get_content",
          "references_for_memory",
          "trace_for_local_slot",
          "attach_qh_record",
          "qh_trace_for_slot",
          "to_dict"
        ],
        "lineno": 148
      }
    ],
    "functions": [
      {
        "name": "tensor_fingerprint",
        "args": [
          "x",
          "max_values"
        ],
        "lineno": 14
      },
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 329
      }
    ],
    "dataclasses": [
      "MirroredContentRule",
      "SharedSlotStoreConfig",
      "SharedSlotWriteResult"
    ]
  },
  "mnemonic_cortex/working_memory/wm_spcp_cross_attention.py": {
    "classes": [
      {
        "name": "WMSPCPCrossAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 16
      },
      {
        "name": "WMSPCPCrossAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 32
      },
      {
        "name": "WMSPCPCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 47
      }
    ],
    "functions": [
      {
        "name": "wm_qd4a_external_memory_contract",
        "args": [],
        "lineno": 99
      }
    ],
    "dataclasses": [
      "WMSPCPCrossAttentionConfig",
      "WMSPCPCrossAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_stability_attention.py": {
    "classes": [
      {
        "name": "WMStabilityAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMStabilityAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 29
      },
      {
        "name": "WMStabilityAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 44
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 94
      }
    ],
    "dataclasses": [
      "WMStabilityAttentionConfig",
      "WMStabilityAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_system_commit_gate.py": {
    "classes": [
      {
        "name": "SystemWriteProposal",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "create",
          "validate",
          "to_trace"
        ],
        "lineno": 22
      },
      {
        "name": "CommitGateDecision",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate",
          "to_dict"
        ],
        "lineno": 108
      },
      {
        "name": "CommitGateEvaluation",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "ok",
          "to_dict"
        ],
        "lineno": 133
      },
      {
        "name": "SystemCommitGate",
        "bases": [],
        "decorators": [],
        "methods": [
          "stage",
          "evaluate",
          "commit",
          "reject",
          "quarantine",
          "rollback_last",
          "trace_summary"
        ],
        "lineno": 167
      }
    ],
    "functions": [
      {
        "name": "wm_qd5a_commit_cortex_contract",
        "args": [],
        "lineno": 483
      }
    ],
    "dataclasses": [
      "SystemWriteProposal",
      "CommitGateDecision",
      "CommitGateEvaluation"
    ]
  },
  "mnemonic_cortex/working_memory/wm_trace.py": {
    "classes": [
      {
        "name": "TraceItem",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 12
      },
      {
        "name": "WMTrace",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "add",
          "merge_dict",
          "set_paamax",
          "update_scores",
          "to_dict",
          "summary"
        ],
        "lineno": 26
      },
      {
        "name": "WMTraceEmitter",
        "bases": [],
        "decorators": [],
        "methods": [
          "start",
          "finish"
        ],
        "lineno": 84
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 103
      }
    ],
    "dataclasses": [
      "TraceItem",
      "WMTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_trace_attention.py": {
    "classes": [
      {
        "name": "WMTraceAttentionConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMTraceAttentionOutput",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "to_dict"
        ],
        "lineno": 28
      },
      {
        "name": "WMTraceAttention",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "forward"
        ],
        "lineno": 43
      }
    ],
    "functions": [
      {
        "name": "wm_qd3a_attention_contract",
        "args": [],
        "lineno": 108
      }
    ],
    "dataclasses": [
      "WMTraceAttentionConfig",
      "WMTraceAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_triplet_state.py": {
    "classes": [
      {
        "name": "WMTripletStateConfig",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "validate"
        ],
        "lineno": 13
      },
      {
        "name": "WMTripletState",
        "bases": [],
        "decorators": [
          "dataclass"
        ],
        "methods": [
          "tensor",
          "shape_summary",
          "to_dict"
        ],
        "lineno": 26
      },
      {
        "name": "WMTripletProjector",
        "bases": [
          "nn.Module"
        ],
        "decorators": [],
        "methods": [
          "project",
          "fuse",
          "forward"
        ],
        "lineno": 53
      }
    ],
    "functions": [
      {
        "name": "wm_qd2a_depth_contract",
        "args": [],
        "lineno": 107
      }
    ],
    "dataclasses": [
      "WMTripletStateConfig",
      "WMTripletState"
    ]
  }
}
```
