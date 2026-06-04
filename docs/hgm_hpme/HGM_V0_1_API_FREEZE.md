# HGM v0.1 API Freeze

This document freezes the additive HGM/HPME public API surface as exported by:

```python
mnemonic_cortex.hypergraph_manifold.__all__
```

## Freeze rule

For HGM v0.1:

1. Existing exported symbols must not be removed without a later migration stage.
2. Existing exported symbols must not be renamed without compatibility aliases.
3. New symbols may be added additively in later stages.
4. QDT/WM internals remain untouched by the HGM v0.1 consolidation.
5. Write-capable runtime integration remains disabled until a later explicit permission stage.

## Frozen module families

- HGM-0A foundation types, enums, validation, traces, tensor contracts.
- HGM-0B probability expansion and scenario candidate extraction.
- HGM-1 hyperedge binding, coherence scoring, conflict and opportunity graphs.
- HGM-2 manifold routing, geometry distances, depth retrieval targets.
- HGM-3 SPCP procedural memory, retrieval, advisory robotics planning.
- HGM-4 QDT/WM bridge payloads and trace-safe memory plans.
- HGM-5 deterministic embeddings and bridge quality scoring.
- HGM-6 write-permission gate and rollback-safe transaction previews.
- HGM-7 simulation-mode write execution scaffold, transaction log, recovery verification.
- HGM-8 runtime embedding scaffold, safe write replay, pipeline benchmark.
- HGM-9 runtime integration evaluation, slot-lattice replay, production-readiness gate.
- HGM-10 release consolidation, API freeze, integration roadmap.

## Runtime status

HGM v0.1 is a stable additive evaluation and integration-preparation release. It is **not** a production write-execution release.

## Generated freeze summary

- Release version: `HGM-v0.1`
- Public symbol count: `176`
- Module count: `56`
- Freeze ID: `hgm10_api_freeze_7d731a644a15dd27`

### Public symbols

- `APIFreezeRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `APIFreezeSymbol` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `ActionPrimitive` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `ActionSequenceBuildResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `BoundScenarioHyperedge` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `BridgeAdapterStatus` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `BridgeEvaluationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `BridgeExecutionPreview` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `BridgePayloadBuildResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `BridgeQualityMetric` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `CoherenceScoreReport` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `CommitReadinessScore` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `ConflictEdge` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `ConflictGraphResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `DepthLayer` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `DepthLayerAssignment` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `DepthRetrievalBridgeResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `DepthRetrievalTarget` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `EmbeddingTrainerOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `EmbeddingTrainerResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `GeometryDistanceResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `GeometryType` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `HGM10ReleaseConsolidationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `HGM10ReleaseOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `HGM1ScenarioGraphResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `HGM2ManifoldRoutingResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `HGM3ProceduralMemoryResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `HGM4BridgeResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `HGM5EmbeddingEvaluationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `HGM6CommitOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `HGM6WritePermissionResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `HGM7ExecutionOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `HGM7WriteExecutionResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `HGM8PipelineEvaluationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `HGM8RuntimeOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `HGM9ReadinessOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `HGM9RuntimeIntegrationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `HGMBridgePayload` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `HGMConfig` — class from `mnemonic_cortex.hypergraph_manifold.config`
- `HGMEmbeddingRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `HyperedgeBindingInput` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `HyperedgeBindingOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `HyperedgeBindingResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `HyperedgeKind` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `HypersetMatrix` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `IntegrationRoadmapItem` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `IntegrationRoadmapRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `IntegrationScore` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `IntegrationScoringResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm5_result`
- `MagnitudeBin` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `ManifoldChart` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `ManifoldRouteAssignment` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `ManifoldRoutingInput` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `ManifoldRoutingOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `ManifoldRoutingResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm2_result`
- `MutationDirection` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `MutationToken` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `NormalizationReport` — class from `mnemonic_cortex.hypergraph_manifold.runtime_result`
- `OpportunityEdge` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `OpportunityGraphResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm1_result`
- `P_VM` — TensorShapeContract from `mnemonic_cortex.hypergraph_manifold.shapes`
- `P_VMD` — TensorShapeContract from `mnemonic_cortex.hypergraph_manifold.shapes`
- `P_VMDC` — TensorShapeContract from `mnemonic_cortex.hypergraph_manifold.shapes`
- `P_VMDCT` — TensorShapeContract from `mnemonic_cortex.hypergraph_manifold.shapes`
- `P_VMDCTA` — TensorShapeContract from `mnemonic_cortex.hypergraph_manifold.shapes`
- `PipelineBenchmarkMetric` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `PipelineBenchmarkResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `ProbabilityExpansionResult` — class from `mnemonic_cortex.hypergraph_manifold.runtime_result`
- `ProbabilityNormalizationMode` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `ProceduralActionSequence` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `ProceduralMemoryRetrievalCandidate` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `ProceduralMemoryRetrievalResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `ProceduralMemoryStoreResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `ProductionReadinessGate` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `QDTRuntimeEvaluationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `QDTRuntimeReadinessMetric` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `QDTWMBridgeOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `QSpinSignature` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `RecoveryVerificationRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `RecoveryVerificationResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `ReleaseConsolidationRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `ReleaseManifestSummary` — class from `mnemonic_cortex.hypergraph_manifold.hgm10_result`
- `RoboticsPlanningActionOption` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `RoboticsPlanningBridgeResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `RollbackManifest` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `RollbackOperation` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `RuntimeEmbeddingRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `RuntimeEmbeddingTrainerResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `SHAPE_CONTRACTS` — dict from `mnemonic_cortex.hypergraph_manifold`
- `SPCPEmbeddingResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `SPCPProceduralOptions` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `SPCPProcedureEmbedding` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `SPCPSimilarityResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm3_result`
- `SafeWriteReplayRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `SafeWriteReplayResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm8_result`
- `ScenarioCandidate` — class from `mnemonic_cortex.hypergraph_manifold.runtime_result`
- `ScenarioExtractionResult` — class from `mnemonic_cortex.hypergraph_manifold.runtime_result`
- `ScenarioHyperedge` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `SharedSlotLatticeHook` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `SharedSlotLatticeHookResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `SlotLatticeReplayBenchmarkResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `SlotLatticeReplayRecord` — class from `mnemonic_cortex.hypergraph_manifold.hgm9_result`
- `TensorShapeContract` — class from `mnemonic_cortex.hypergraph_manifold.shapes`
- `TraceEventKind` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `TraceRecord` — class from `mnemonic_cortex.hypergraph_manifold.types`
- `TraceSafeMemoryPlan` — class from `mnemonic_cortex.hypergraph_manifold.hgm4_result`
- `TransactionCommitPreview` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `TransactionLog` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `TransactionLogEntry` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `TransactionOperationPreview` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `ValidationMessage` — class from `mnemonic_cortex.hypergraph_manifold.validation`
- `ValidationResult` — class from `mnemonic_cortex.hypergraph_manifold.validation`
- `ValidationSeverity` — class from `mnemonic_cortex.hypergraph_manifold.enums`
- `WriteExecutionAdapterStatus` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `WriteExecutionResult` — class from `mnemonic_cortex.hypergraph_manifold.hgm7_result`
- `WritePermissionState` — class from `mnemonic_cortex.hypergraph_manifold.hgm6_result`
- `assign_depth_retrieval_targets` — function from `mnemonic_cortex.hypergraph_manifold.depth_retrieval`
- `benchmark_hgm_pipeline` — function from `mnemonic_cortex.hypergraph_manifold.pipeline_benchmark`
- `benchmark_slot_lattice_replay` — function from `mnemonic_cortex.hypergraph_manifold.slot_lattice_replay_benchmark`
- `bind_scenario_candidates` — function from `mnemonic_cortex.hypergraph_manifold.hyperedge_binder`
- `build_action_sequence_from_hgm2_route` — function from `mnemonic_cortex.hypergraph_manifold.action_sequence`
- `build_baseline_hgm_embeddings` — function from `mnemonic_cortex.hypergraph_manifold.embedding_trainer`
- `build_bridge_payload_from_hgm_record` — function from `mnemonic_cortex.hypergraph_manifold.bridge_payloads`
- `build_hgm10_release_consolidation` — function from `mnemonic_cortex.hypergraph_manifold.hgm10_pipeline`
- `build_hgm1_scenario_graph` — function from `mnemonic_cortex.hypergraph_manifold.hyperedge_binder`
- `build_hgm2_manifold_routing` — function from `mnemonic_cortex.hypergraph_manifold.manifold_router`
- `build_hgm3_spcp_procedural_memory` — function from `mnemonic_cortex.hypergraph_manifold.robotics_planning_bridge`
- `build_hgm4_qdt_wm_bridge` — function from `mnemonic_cortex.hypergraph_manifold.qdt_wm_bridge`
- `build_hgm5_embedding_evaluation` — function from `mnemonic_cortex.hypergraph_manifold.integration_scoring`
- `build_hgm6_write_permission_gate` — function from `mnemonic_cortex.hypergraph_manifold.write_permission_gate`
- `build_hgm7_write_execution_adapter` — function from `mnemonic_cortex.hypergraph_manifold.write_execution_adapter`
- `build_hgm8_pipeline_evaluation` — function from `mnemonic_cortex.hypergraph_manifold.pipeline_benchmark`
- `build_hgm9_runtime_integration_evaluation` — function from `mnemonic_cortex.hypergraph_manifold.hgm9_pipeline`
- `build_hgm_integration_roadmap` — function from `mnemonic_cortex.hypergraph_manifold.integration_roadmap`
- `build_probability_expansion` — function from `mnemonic_cortex.hypergraph_manifold.probability_expander`
- `build_robotics_planning_options` — function from `mnemonic_cortex.hypergraph_manifold.robotics_planning_bridge`
- `build_rollback_manifest` — function from `mnemonic_cortex.hypergraph_manifold.rollback_plan`
- `build_runtime_embedding_trainer` — function from `mnemonic_cortex.hypergraph_manifold.runtime_embedding_trainer`
- `build_shared_slot_lattice_hooks` — function from `mnemonic_cortex.hypergraph_manifold.slot_lattice_hooks`
- `build_trace_safe_memory_plan` — function from `mnemonic_cortex.hypergraph_manifold.trace_safe_memory_integration`
- `build_transaction_commit_preview` — function from `mnemonic_cortex.hypergraph_manifold.transaction_preview`
- `build_transaction_log` — function from `mnemonic_cortex.hypergraph_manifold.transaction_log`
- `build_transaction_operation_previews` — function from `mnemonic_cortex.hypergraph_manifold.transaction_preview`
- `build_write_execution_adapter_status` — function from `mnemonic_cortex.hypergraph_manifold.write_execution_adapter`
- `build_write_permission_state` — function from `mnemonic_cortex.hypergraph_manifold.write_permission_gate`
- `compute_geometry_distance` — function from `mnemonic_cortex.hypergraph_manifold.geometry_distance`
- `compute_spcp_procedure_embedding` — function from `mnemonic_cortex.hypergraph_manifold.spcp_procedural`
- `consolidate_hgm_release_manifests` — function from `mnemonic_cortex.hypergraph_manifold.release_consolidation`
- `contract_by_name` — function from `mnemonic_cortex.hypergraph_manifold.shapes`
- `detect_conflict_edges` — function from `mnemonic_cortex.hypergraph_manifold.conflict_graph`
- `detect_opportunity_edges` — function from `mnemonic_cortex.hypergraph_manifold.opportunity_graph`
- `detect_qdt_wm_adapter_status` — function from `mnemonic_cortex.hypergraph_manifold.qdt_wm_bridge`
- `evaluate_bridge_payload_quality` — function from `mnemonic_cortex.hypergraph_manifold.quality_metrics`
- `evaluate_execution_preview_quality` — function from `mnemonic_cortex.hypergraph_manifold.quality_metrics`
- `evaluate_qdt_runtime_integration` — function from `mnemonic_cortex.hypergraph_manifold.qdt_runtime_evaluation`
- `evaluate_safe_write_replay` — function from `mnemonic_cortex.hypergraph_manifold.safe_write_replay`
- `evaluate_slot_hook_quality` — function from `mnemonic_cortex.hypergraph_manifold.quality_metrics`
- `execute_write_adapter` — function from `mnemonic_cortex.hypergraph_manifold.write_execution_adapter`
- `expand_from_mutation_tokens` — function from `mnemonic_cortex.hypergraph_manifold.probability_expander`
- `extract_top_k_scenarios` — function from `mnemonic_cortex.hypergraph_manifold.scenario_extraction`
- `freeze_hgm_public_api` — function from `mnemonic_cortex.hypergraph_manifold.api_freeze`
- `infer_probability_shape_contract` — function from `mnemonic_cortex.hypergraph_manifold.probability_expander`
- `normalize_probability_payload` — function from `mnemonic_cortex.hypergraph_manifold.normalization`
- `preview_bridge_execution` — function from `mnemonic_cortex.hypergraph_manifold.trace_safe_memory_integration`
- `retrieve_similar_procedures` — function from `mnemonic_cortex.hypergraph_manifold.procedural_memory`
- `route_hyperedges_to_manifold_charts` — function from `mnemonic_cortex.hypergraph_manifold.manifold_router`
- `score_commit_readiness` — function from `mnemonic_cortex.hypergraph_manifold.commit_readiness`
- `score_hgm_integration_readiness` — function from `mnemonic_cortex.hypergraph_manifold.integration_scoring`
- `score_hyperedge_coherence` — function from `mnemonic_cortex.hypergraph_manifold.coherence`
- `score_production_readiness` — function from `mnemonic_cortex.hypergraph_manifold.production_readiness_gate`
- `spcp_procedure_similarity` — function from `mnemonic_cortex.hypergraph_manifold.spcp_procedural`
- `store_procedural_sequences` — function from `mnemonic_cortex.hypergraph_manifold.procedural_memory`
- `validate_action_primitive` — function from `mnemonic_cortex.hypergraph_manifold.action_sequence`
- `validate_action_sequence` — function from `mnemonic_cortex.hypergraph_manifold.action_sequence`
- `validate_probability_payload` — function from `mnemonic_cortex.hypergraph_manifold.probability_expander`
- `verify_recovery` — function from `mnemonic_cortex.hypergraph_manifold.recovery_verification`
