# WM-7A API Surface Audit

## Token budget / selected scope

```json
{
  "target_scope": "Create final release/readiness pack for QDT-WM-MAAE.",
  "minimum_complete_version": [
    "API surface audit",
    "module dependency map",
    "benchmark harness",
    "full pytest run",
    "benchmark run",
    "release manifest",
    "production readiness document",
    "tracker/deferred updates",
    "final adequacy/ship-check"
  ],
  "deep_implementation_version": [
    "AST-based API audit",
    "AST-based dependency map",
    "executable smoke benchmark harness",
    "full pytest output captured",
    "benchmark JSON/markdown captured",
    "release manifest",
    "production caveats and real-source integration requirements stated honestly",
    "tracker/deferred register updated"
  ],
  "estimated_file_module_count": 50,
  "expected_doc_count": 8,
  "expected_test_count": 28,
  "benchmark_count": 3,
  "clean_split_points": [
    "WM-7A.1 API surface audit and release manifest",
    "WM-7A.2 benchmark harness and results",
    "WM-7A.3 production readiness and final tracker closure"
  ],
  "selected_split_scope": "All WM-7A deliverables in files; response prints summary only.",
  "explicit_out_of_scope_items": [
    "Patching real EnhancedMnemonicCortex source file because it is not present in this pack",
    "Persistent database/storage backend",
    "Real LTM/MANN/SPCP external adapters",
    "Real quantum/holographic hardware backend",
    "Long-running hardware performance profiling"
  ]
}
```

## Public API by module

| module | classes | functions | dataclasses |
| --- | --- | --- | --- |
| mnemonic_cortex/working_memory/__init__.py |  |  |  |
| mnemonic_cortex/working_memory/_wm_light_transformer.py | LightweightTransformerBlock, LightweightTransformerStack |  |  |
| mnemonic_cortex/working_memory/bounded_associative_spread.py | BoundedAssociativeSpreadConfig, BoundedSpreadTrace, BoundedAssociativeSpread |  | BoundedAssociativeSpreadConfig, BoundedSpreadTrace |
| mnemonic_cortex/working_memory/context_depth_adapter.py | ContextDepthAdapter |  |  |
| mnemonic_cortex/working_memory/context_geometry_maps.py | ContextGeometryMap | build_default_context_geometry_maps, validate_context_geometry_map |  |
| mnemonic_cortex/working_memory/context_map_selector.py | ContextSelectionTrace, ContextMapSelector |  | ContextSelectionTrace |
| mnemonic_cortex/working_memory/context_stability_guard.py | ContextStabilityReport, ContextStabilityGuard |  | ContextStabilityReport |
| mnemonic_cortex/working_memory/context_to_wm_bridge.py | ContextToWMBridge |  |  |
| mnemonic_cortex/working_memory/context_trace.py | ContextMountTrace |  | ContextMountTrace |
| mnemonic_cortex/working_memory/context_triplet_projector.py | ContextTripletProjector |  |  |
| mnemonic_cortex/working_memory/curvature_metric_policy.py | CurvatureMetricPolicyConfig, CurvatureMetricPolicyOutput, CurvatureMetricPolicy |  | CurvatureMetricPolicyConfig, CurvatureMetricPolicyOutput |
| mnemonic_cortex/working_memory/curved_local_trace.py | CurvedTraceEvent, CurvedLocalTrace, CurvedLocalTraceBuilder |  | CurvedTraceEvent, CurvedLocalTrace |
| mnemonic_cortex/working_memory/curved_resonant_wm_core.py | ResonanceStepTrace, CurvedResonanceTrace, CurvedResonanceConfig, CurvedResonantWMCore |  | ResonanceStepTrace, CurvedResonanceTrace, CurvedResonanceConfig |
| mnemonic_cortex/working_memory/curved_shadow_write.py | CurvedShadowWriteConfig, ShadowWriteProposal, ShadowWriteDecision, CurvedShadowWriteBuffer |  | CurvedShadowWriteConfig, ShadowWriteProposal, ShadowWriteDecision |
| mnemonic_cortex/working_memory/curved_slot_state.py | CurvedSlotStateConfig, CurvedSlotSnapshot, CurvedSlotStateTrace, CurvedSlotStateBank |  | CurvedSlotStateConfig, CurvedSlotSnapshot, CurvedSlotStateTrace |
| mnemonic_cortex/working_memory/depth_specific_addressing.py | DepthSpecificAddressingConfig, DepthSpecificAddressingTrace, DepthSpecificAddressingOutput, DepthSpecificAddressing |  | DepthSpecificAddressingConfig, DepthSpecificAddressingTrace, DepthSpecificAddressingOutput |
| mnemonic_cortex/working_memory/geometry_aware_addressing.py | GeometryAwareAddressingConfig, GeometryAwareAddressingTrace, GeometryAwareAddressingOutput, GeometryAwareAddressing |  | GeometryAwareAddressingConfig, GeometryAwareAddressingTrace, GeometryAwareAddressingOutput |
| mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py | CurvedMemoryReadTrace, EnhancedCurvedMemory |  | CurvedMemoryReadTrace |
| mnemonic_cortex/working_memory/qdt_working_memory.py | QDTWorkingMemory |  |  |
| mnemonic_cortex/working_memory/wm_compatibility_wrapper.py | QDTWMCompatibilityConfig, QDTWMCompatibilityTrace, QDTWMCompatibilityWrapper |  | QDTWMCompatibilityConfig, QDTWMCompatibilityTrace |
| mnemonic_cortex/working_memory/wm_config.py | QDTWorkingMemoryConfig |  | QDTWorkingMemoryConfig |
| mnemonic_cortex/working_memory/wm_conflict_attention.py | WMConflictAttentionConfig, WMConflictAttentionOutput, WMConflictAttention |  | WMConflictAttentionConfig, WMConflictAttentionOutput |
| mnemonic_cortex/working_memory/wm_context_mount.py | GeometryMountedContextBuffer |  |  |
| mnemonic_cortex/working_memory/wm_cortex_integration.py | CortexWorkingMemoryIntegrationConfig, CortexWorkingMemoryMigrationResult, EnhancedMnemonicCortexQDTAdapter | build_qdt_working_memory_for_cortex, replace_cortex_working_memory, migration_patch_template | CortexWorkingMemoryIntegrationConfig, CortexWorkingMemoryMigrationResult |
| mnemonic_cortex/working_memory/wm_counterfactual_attention.py | WMCounterfactualAttentionConfig, WMCounterfactualAttentionOutput, WMCounterfactualAttention |  | WMCounterfactualAttentionConfig, WMCounterfactualAttentionOutput |
| mnemonic_cortex/working_memory/wm_cross_depth_transformer.py | WMCrossDepthTransformerConfig, WMCrossDepthTransformerTrace, WMCrossDepthTransformer |  | WMCrossDepthTransformerConfig, WMCrossDepthTransformerTrace |
| mnemonic_cortex/working_memory/wm_curved_core.py | WMCurvedAssociativeCore |  |  |
| mnemonic_cortex/working_memory/wm_depth_adapters.py | WMDepthAdaptersConfig, WMDepthAdaptersTrace, WMDepthAdapters |  | WMDepthAdaptersConfig, WMDepthAdaptersTrace |
| mnemonic_cortex/working_memory/wm_depth_fusion.py | WMDepthFusionConfig, WMDepthFusionTrace, WMDepthFusion |  | WMDepthFusionConfig, WMDepthFusionTrace |
| mnemonic_cortex/working_memory/wm_dual_fusion.py | WMDualFusionConfig, WMDualFusionOutput, WMDualFusionController |  | WMDualFusionConfig, WMDualFusionOutput |
| mnemonic_cortex/working_memory/wm_evidence_attention.py | WMEvidenceAttentionConfig, WMEvidenceAttentionOutput, WMEvidenceAttention |  | WMEvidenceAttentionConfig, WMEvidenceAttentionOutput |
| mnemonic_cortex/working_memory/wm_external_memory_interfaces.py | ExternalMemoryQuery, ExternalMemoryResponse, SyntheticExternalMemoryBank |  | ExternalMemoryQuery, ExternalMemoryResponse |
| mnemonic_cortex/working_memory/wm_geometry_linker.py | GeometryLink, WMGeometryLinkerConfig, WMGeometryLinker |  | GeometryLink, WMGeometryLinkerConfig |
| mnemonic_cortex/working_memory/wm_geometry_scoring.py | WMGeometryScoringConfig, WMGeometryScoringOutput, WMGeometryScoring |  | WMGeometryScoringConfig, WMGeometryScoringOutput |
| mnemonic_cortex/working_memory/wm_intra_depth_transformer.py | WMIntraDepthTransformerConfig, WMIntraDepthTransformerTrace, WMIntraDepthTransformer |  | WMIntraDepthTransformerConfig, WMIntraDepthTransformerTrace |
| mnemonic_cortex/working_memory/wm_ltm_cross_attention.py | WMLTMCrossAttentionConfig, WMLTMCrossAttentionOutput, WMLTMCrossAttention |  | WMLTMCrossAttentionConfig, WMLTMCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_mann_cross_attention.py | WMMANNCrossAttentionConfig, WMMANNTraceVisibility, WMMANNCrossAttentionOutput, WMMANNCrossAttention |  | WMMANNCrossAttentionConfig, WMMANNTraceVisibility, WMMANNCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_memory_augmented_attention.py | WMMemoryAugmentedAttentionConfig, WMMemoryAugmentedAttentionOutput, WMMemoryAugmentedAttention |  | WMMemoryAugmentedAttentionConfig, WMMemoryAugmentedAttentionOutput |
| mnemonic_cortex/working_memory/wm_novelty_attention.py | WMNoveltyAttentionConfig, WMNoveltyAttentionOutput, WMNoveltyAttention |  | WMNoveltyAttentionConfig, WMNoveltyAttentionOutput |
| mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py | QHCodeSchema, QHInterferenceReport, QHStorageRecord, QuantumHolographicStorageConfig, QuantumHolographicStorage | build_qh_code_schema | QHInterferenceReport, QHStorageRecord, QuantumHolographicStorageConfig |
| mnemonic_cortex/working_memory/wm_quaternion_depth.py | QuaternionDepthTrace, QuaternionDepthConfig, QuaternionDepthReplicator | normalize_quaternion, quaternion_conjugate, quaternion_multiply, rotate_vectors_by_quaternion | QuaternionDepthTrace, QuaternionDepthConfig |
| mnemonic_cortex/working_memory/wm_retrieval_lanes.py | RetrievalLaneConfig, RetrievalLaneOutput, WMRetrievalLanesOutput, WMRetrievalLanes |  | RetrievalLaneConfig, RetrievalLaneOutput, WMRetrievalLanesOutput |
| mnemonic_cortex/working_memory/wm_shared_slot_registry.py | SharedSlotMirrorRef, SharedSlotRecord, SharedSlotRegistry | canonical_slot_id | SharedSlotMirrorRef, SharedSlotRecord |
| mnemonic_cortex/working_memory/wm_shared_slot_store.py | MirroredContentRule, SharedSlotStoreConfig, SharedSlotWriteResult, SharedSlotStore | tensor_fingerprint | MirroredContentRule, SharedSlotStoreConfig, SharedSlotWriteResult |
| mnemonic_cortex/working_memory/wm_spcp_cross_attention.py | WMSPCPCrossAttentionConfig, WMSPCPCrossAttentionOutput, WMSPCPCrossAttention |  | WMSPCPCrossAttentionConfig, WMSPCPCrossAttentionOutput |
| mnemonic_cortex/working_memory/wm_stability_attention.py | WMStabilityAttentionConfig, WMStabilityAttentionOutput, WMStabilityAttention |  | WMStabilityAttentionConfig, WMStabilityAttentionOutput |
| mnemonic_cortex/working_memory/wm_system_commit_gate.py | SystemWriteProposal, CommitGateDecision, CommitGateEvaluation, SystemCommitGate |  | SystemWriteProposal, CommitGateDecision, CommitGateEvaluation |
| mnemonic_cortex/working_memory/wm_trace.py | TraceItem, WMTrace, WMTraceEmitter |  | TraceItem, WMTrace |
| mnemonic_cortex/working_memory/wm_trace_attention.py | WMTraceAttentionConfig, WMTraceAttentionOutput, WMTraceAttention |  | WMTraceAttentionConfig, WMTraceAttentionOutput |
| mnemonic_cortex/working_memory/wm_triplet_state.py | WMTripletStateConfig, WMTripletState, WMTripletProjector |  | WMTripletStateConfig, WMTripletState |

## Raw API audit JSON

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
        "methods": [
          "forward"
        ],
        "decorators": []
      },
      {
        "name": "LightweightTransformerStack",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "BoundedSpreadTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "BoundedAssociativeSpread",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "transition_matrix",
          "forward",
          "hebbian_update",
          "validate_transition"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "forward"
        ],
        "decorators": []
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
        "methods": [
          "normalized_for_depths"
        ],
        "decorators": [
          "dataclass(frozen=True)"
        ]
      }
    ],
    "functions": [
      {
        "name": "build_default_context_geometry_maps",
        "args": [
          "num_depths"
        ]
      },
      {
        "name": "validate_context_geometry_map",
        "args": [
          "map_spec",
          "num_depths"
        ]
      }
    ],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/context_map_selector.py": {
    "classes": [
      {
        "name": "ContextSelectionTrace",
        "bases": [],
        "methods": [],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "ContextMapSelector",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "select"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": [
      "ContextSelectionTrace"
    ]
  },
  "mnemonic_cortex/working_memory/context_stability_guard.py": {
    "classes": [
      {
        "name": "ContextStabilityReport",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "ContextStabilityGuard",
        "bases": [],
        "methods": [
          "check",
          "repair"
        ],
        "decorators": []
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
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/context_trace.py": {
    "classes": [
      {
        "name": "ContextMountTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
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
        "methods": [
          "project",
          "fuse",
          "forward"
        ],
        "decorators": []
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvatureMetricPolicyOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvatureMetricPolicy",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "clamped_components",
          "drift_penalty",
          "set_reference_to_current",
          "context_conditioned_curvature",
          "forward",
          "repair_in_place",
          "validate_policy"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedLocalTrace",
        "bases": [],
        "methods": [
          "add_event",
          "set_write_decision",
          "merge_paamax",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedLocalTraceBuilder",
        "bases": [],
        "methods": [
          "from_resonance_trace",
          "with_geometry_map",
          "with_curvature_state",
          "with_depth_contribution",
          "with_disagreement",
          "build"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedResonanceTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedResonanceConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedResonantWMCore",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "ShadowWriteProposal",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "ShadowWriteDecision",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedShadowWriteBuffer",
        "bases": [],
        "methods": [
          "stage",
          "interference_score",
          "evaluate",
          "commit",
          "reject",
          "pending_count",
          "to_dict"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedSlotSnapshot",
        "bases": [],
        "methods": [
          "shape_summary"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedSlotStateTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CurvedSlotStateBank",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "stable_tensors",
          "repair_in_place",
          "validate_state",
          "snapshot",
          "update_slots",
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "DepthSpecificAddressingTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "DepthSpecificAddressingOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "DepthSpecificAddressing",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "GeometryAwareAddressingTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "GeometryAwareAddressingOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "GeometryAwareAddressing",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "EnhancedCurvedMemory",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "enable_energy_efficient_mode",
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_compatibility_wrapper.py": {
    "classes": [
      {
        "name": "QDTWMCompatibilityConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QDTWMCompatibilityTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QDTWMCompatibilityWrapper",
        "bases": [
          "nn.Module"
        ],
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
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMConflictAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMConflictAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "select_map",
          "mount"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_cortex_integration.py": {
    "classes": [
      {
        "name": "CortexWorkingMemoryIntegrationConfig",
        "bases": [],
        "methods": [
          "validate",
          "qdt_config",
          "compatibility_config"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CortexWorkingMemoryMigrationResult",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "EnhancedMnemonicCortexQDTAdapter",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "read_working_memory",
          "process_working_memory",
          "write_working_memory",
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [
      {
        "name": "build_qdt_working_memory_for_cortex",
        "args": [
          "config"
        ]
      },
      {
        "name": "replace_cortex_working_memory",
        "args": [
          "cortex",
          "config"
        ]
      },
      {
        "name": "migration_patch_template",
        "args": [
          "config"
        ]
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMCounterfactualAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMCounterfactualAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMCrossDepthTransformerTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMCrossDepthTransformer",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "last_trace",
          "enable_energy_efficient_mode",
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": []
  },
  "mnemonic_cortex/working_memory/wm_depth_adapters.py": {
    "classes": [
      {
        "name": "WMDepthAdaptersConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDepthAdaptersTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDepthAdapters",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDepthFusionTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDepthFusion",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": [
      "WMDepthFusionConfig",
      "WMDepthFusionTrace"
    ]
  },
  "mnemonic_cortex/working_memory/wm_dual_fusion.py": {
    "classes": [
      {
        "name": "WMDualFusionConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDualFusionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMDualFusionController",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMEvidenceAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMEvidenceAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": [
      "WMEvidenceAttentionConfig",
      "WMEvidenceAttentionOutput"
    ]
  },
  "mnemonic_cortex/working_memory/wm_external_memory_interfaces.py": {
    "classes": [
      {
        "name": "ExternalMemoryQuery",
        "bases": [],
        "methods": [
          "validate",
          "to_trace"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "ExternalMemoryResponse",
        "bases": [],
        "methods": [
          "validate",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SyntheticExternalMemoryBank",
        "bases": [],
        "methods": [
          "query"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": [
      "ExternalMemoryQuery",
      "ExternalMemoryResponse"
    ]
  },
  "mnemonic_cortex/working_memory/wm_geometry_linker.py": {
    "classes": [
      {
        "name": "GeometryLink",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMGeometryLinkerConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMGeometryLinker",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "lane_bias",
          "to_trace"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMGeometryScoringOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMGeometryScoring",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMIntraDepthTransformerTrace",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMIntraDepthTransformer",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMLTMCrossAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMLTMCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMMANNTraceVisibility",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMMANNCrossAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMMANNCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMMemoryAugmentedAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMMemoryAugmentedAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMNoveltyAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMNoveltyAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "to_tuple",
          "composite_code",
          "to_dict"
        ],
        "decorators": [
          "dataclass(frozen=True)"
        ]
      },
      {
        "name": "QHInterferenceReport",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QHStorageRecord",
        "bases": [],
        "methods": [
          "validate",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QuantumHolographicStorageConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QuantumHolographicStorage",
        "bases": [],
        "methods": [
          "check_interference",
          "create_record",
          "create_from_shared_slot",
          "trace_summary",
          "to_dict"
        ],
        "decorators": []
      }
    ],
    "functions": [
      {
        "name": "build_qh_code_schema",
        "args": []
      }
    ],
    "dataclasses": [
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
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QuaternionDepthConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "QuaternionDepthReplicator",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "normalized_depth_quaternions",
          "replicate",
          "forward",
          "depth_consistency_report",
          "set_depth_quaternion",
          "dual_quaternion_status"
        ],
        "decorators": []
      }
    ],
    "functions": [
      {
        "name": "normalize_quaternion",
        "args": [
          "q",
          "eps"
        ]
      },
      {
        "name": "quaternion_conjugate",
        "args": [
          "q"
        ]
      },
      {
        "name": "quaternion_multiply",
        "args": [
          "a",
          "b"
        ]
      },
      {
        "name": "rotate_vectors_by_quaternion",
        "args": [
          "v",
          "q",
          "eps"
        ]
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "RetrievalLaneOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMRetrievalLanesOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMRetrievalLanes",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward",
          "stability_report"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SharedSlotRecord",
        "bases": [],
        "methods": [
          "validate",
          "add_or_update_mirror",
          "set_conflict",
          "grant_write_permission",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SharedSlotRegistry",
        "bases": [],
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
        "decorators": []
      }
    ],
    "functions": [
      {
        "name": "canonical_slot_id",
        "args": [
          "namespace",
          "local_slot_id",
          "content_fingerprint"
        ]
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
        "methods": [
          "validate",
          "attach_qh_record",
          "qh_trace_for_slot",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SharedSlotStoreConfig",
        "bases": [],
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SharedSlotWriteResult",
        "bases": [],
        "methods": [
          "attach_qh_record",
          "qh_trace_for_slot",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SharedSlotStore",
        "bases": [],
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
        "decorators": []
      }
    ],
    "functions": [
      {
        "name": "tensor_fingerprint",
        "args": [
          "x",
          "max_values"
        ]
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMSPCPCrossAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMSPCPCrossAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMStabilityAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMStabilityAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "create",
          "validate",
          "to_trace"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CommitGateDecision",
        "bases": [],
        "methods": [
          "validate",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "CommitGateEvaluation",
        "bases": [],
        "methods": [
          "ok",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "SystemCommitGate",
        "bases": [],
        "methods": [
          "stage",
          "evaluate",
          "commit",
          "reject",
          "quarantine",
          "rollback_last",
          "trace_summary"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTrace",
        "bases": [],
        "methods": [
          "add",
          "merge_dict",
          "set_paamax",
          "update_scores",
          "to_dict",
          "summary"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTraceEmitter",
        "bases": [],
        "methods": [
          "start",
          "finish"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTraceAttentionOutput",
        "bases": [],
        "methods": [
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTraceAttention",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
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
        "methods": [
          "validate"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTripletState",
        "bases": [],
        "methods": [
          "tensor",
          "shape_summary",
          "to_dict"
        ],
        "decorators": [
          "dataclass"
        ]
      },
      {
        "name": "WMTripletProjector",
        "bases": [
          "nn.Module"
        ],
        "methods": [
          "project",
          "fuse",
          "forward"
        ],
        "decorators": []
      }
    ],
    "functions": [],
    "dataclasses": [
      "WMTripletStateConfig",
      "WMTripletState"
    ]
  }
}
```
