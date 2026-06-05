# QDT-WM-MAAE Patch and Upgrade Tracker — WM-0B Update

| tracker_id | stage/split | affected area | issue | severity | status | required action | downstream impact | completion evidence |
|---|---|---|---|---|---|---|---|---|
| QDT-WM-UPG-WM0B-0001 | WM-0B | Context maps | Full context map preset family needed beyond WM-0A.3 skeleton | blocker | complete | Implement 10 context maps | Enables geometry-mounted context buffer | context_geometry_maps.py |
| QDT-WM-UPG-WM0B-0002 | WM-0B | Context selector | Task/context hint selector needed | high | complete | Implement ContextMapSelector | Enables reasoning-specific context maps | context_map_selector.py |
| QDT-WM-UPG-WM0B-0003 | WM-0B | Triplets | Context triplet projection needed | high | complete | Implement ContextTripletProjector | Preserves dimensional-depth triplet doctrine | context_triplet_projector.py |
| QDT-WM-UPG-WM0B-0004 | WM-0B | Mounting | Context-to-WM bridge for [B,Z,T,3,D] needed | blocker | complete | Implement ContextToWMBridge | Enables depth replica mounting | context_to_wm_bridge.py |
| QDT-WM-UPG-WM0B-0005 | WM-0B | PAAMA-X | Policy/governance map and tags needed | high | complete | Add policy_governance map and PAAMA-X tags | Enables policy lane compatibility | context maps + tests |
| QDT-WM-UPG-WM0B-0006 | WM-0B | QH storage | Quantum-holographic context map needed | high | complete | Add quantum_holographic map | Preserves QH storage prep | context maps + tests |

| QDT-WM-UPG-WM0B-PATCH-0001 | WM-0B | __init__ exports | Context classes were not exported from package root, causing import test failure | blocker | complete | Patch __init__.py export surface and rerun tests | Required for import compatibility | Patched and rerun in WM-0B |

| QDT-WM-UPG-WM0B-PATCH-0002 | WM-0B | frozen context map trace attachment | ContextGeometryMap is frozen, so direct trace assignment failed | blocker | complete | Use object.__setattr__ compatibility attachment | Required for backward API test | Patched and rerun in WM-0B |

| QDT-WM-UPG-WM1A-0001 | WM-1A | Curved WM preservation | Canonical-compatible EnhancedCurvedMemory required because no live repo file is mounted | blocker | complete | Implement legacy_enhanced_curved_memory.py and delegate wrapper | Preserves curved WM identity | Source/tests created |
| QDT-WM-UPG-WM1A-0002 | WM-1A | Wrapper delegation | WMCurvedAssociativeCore must delegate to supplied canonical module when available | blocker | complete | Patch wm_curved_core.py | Enables real canonical module injection later | Test added |
| QDT-WM-UPG-WM1A-0003 | WM-1A | Preservation tests | Core curved attributes and read/write/process behavior need tests | high | complete | Add test_wm1a_curved_preservation.py | Prevents accidental replacement | Tests pass |

| QDT-WM-UPG-WM1A-PATCH-0001 | WM-1A | package exports | Broad optional import block suppressed WMCurvedAssociativeCore export | blocker | complete | Rebuild __init__.py with deterministic core exports and isolated optional imports | Required for preservation tests | Patched and rerun |

| QDT-WM-UPG-WM1B-0001 | WM-1B | Curved Resonant WM | Curved Resonant WM Core needed around preserved inner core | blocker | complete | Implement curved_resonant_wm_core.py | Adds resonance without replacing curved WM | Source/tests created |
| QDT-WM-UPG-WM1B-0002 | WM-1B | Bounded resonance | Resonance steps must be bounded | high | complete | Enforce min(requested, max) | Prevents runaway loops | Test added |
| QDT-WM-UPG-WM1B-0003 | WM-1B | PAAMA-X trace metadata | Resonance trace must expose governance metadata | high | complete | Add paamax_metadata | Enables later governance | Test added |

| QDT-WM-UPG-WM1C-0001 | WM-1C | CurvedSlotState | Curved slot bank needed with content/position/tangent/phase/curvature/importance/confidence/trace | blocker | complete | Implement curved_slot_state.py | Required for WM-1D addressing | Source/tests created |
| QDT-WM-UPG-WM1C-0002 | WM-1C | CurvatureMetricPolicy | Global/per-slot/per-depth/context curvature policy needed | blocker | complete | Implement curvature_metric_policy.py | Required for geometry-aware addressing | Source/tests created |
| QDT-WM-UPG-WM1C-0003 | WM-1C | Stability | Slot and curvature values require clamps/repair | high | complete | Add repair/validate methods | Prevents invalid curvature/position state | Tests pass |

| QDT-WM-UPG-WM1D-0001 | WM-1D | Geometry-aware addressing | Addressing needed content, curved distance, phase, importance, confidence, trace reliability, context bias, and curvature policy | blocker | complete | Implement geometry_aware_addressing.py | Enables stronger curved WM reads | Source/tests created |
| QDT-WM-UPG-WM1D-0002 | WM-1D | Bounded associative spread | Spread needed row-stochastic, sparse, spectral-clamped, bounded update | blocker | complete | Implement bounded_associative_spread.py | Prevents noisy/runaway spread | Source/tests created |
| QDT-WM-UPG-WM1D-0003 | WM-1D | Hebbian update | Curved associative strengthening needed bounded update | high | complete | Add hebbian_update | Enables safe future adaptation | Tests created |
| QDT-WM-UPG-WM1D-0004 | WM-1D | Resonant core integration | WM-1D addressing/spread should enrich CurvedResonantWMCore | high | complete | Patch optional integration hooks | Allows use without replacing inner core | Tests created |

| QDT-WM-UPG-WM1D-PATCH-0001 | WM-1D | CurvedResonantWMCore integration | Resonance seed hidden_dim could mismatch addressing input_dim | blocker | complete | Add compatible query bridge and skip seed nudge on dimension mismatch | Required for optional WM-1D integration | Patched and tests rerun |

| QDT-WM-UPG-WM1E-0001 | WM-1E | Curved local trace | Local trace schema needed selected slots, activation route, curvature, geometry, depth, confidence, novelty, disagreement, write decision | blocker | complete | Implement curved_local_trace.py | Enables WM-local PAAMA-X traceability | Source/tests created |
| QDT-WM-UPG-WM1E-0002 | WM-1E | Curved shadow writes | Writes must stage before commit | blocker | complete | Implement curved_shadow_write.py | Preserves simultaneous read/write doctrine | Source/tests created |
| QDT-WM-UPG-WM1E-0003 | WM-1E | Resonant core write path | CurvedResonantWMCore needed shadow-write integration | high | complete | Patch write path | Enables staged writes | Tests created |

| QDT-WM-UPG-WM2A-0001 | WM-2A | Quaternion depth replication | Scalar-only depth modulation needed replacement with true packed 3D quaternion rotations | blocker | complete | Patch wm_quaternion_depth.py | Enables real dimensional-depth rotations | Source/tests created |
| QDT-WM-UPG-WM2A-0002 | WM-2A | Remainder dimensions | D not divisible by 3 must be handled safely | high | complete | Preserve remainder dimensions | Prevents feature loss | Tests pass |
| QDT-WM-UPG-WM2A-0003 | WM-2A | Dual quaternion hook | Spatial SE(3) hook needed but must not be fake-completed | medium | complete | Add explicit placeholder/status API | Preserves roadmap honesty | dual_quaternion_status test |

| QDT-WM-UPG-WM2B-0001 | WM-2B | Source-integrity gate | Several later-stage modules missing from live tree | blocker | complete | Audit module presence and create WM-2B required modules only | Prevents fake completion | Audit recorded in 62_wm2b doc |
| QDT-WM-UPG-WM2B-0002 | WM-2B | Intra-depth transformer | Need transformer over [B,Z,T,3,D] within each depth/triplet stream | blocker | complete | Implement wm_intra_depth_transformer.py | Enables QDT temporal cognition | Source/tests created |
| QDT-WM-UPG-WM2B-0003 | WM-2B | Cross-depth transformer | Need transformer exchange across depth slices | blocker | complete | Implement wm_cross_depth_transformer.py | Enables cross-depth cognition | Source/tests created |
| QDT-WM-UPG-WM2B-0004 | WM-2B | Depth-specific addressing | Need [B,Z,S] slot activation using curvature and context geometry | blocker | complete | Implement depth_specific_addressing.py | Enables depth-aware WM retrieval | Source/tests created |
| QDT-WM-UPG-WM2B-0005 | WM-2B | Deferred module restoration | WM trace/triplet/depth fusion/QDT assembly modules remain missing | high | incomplete | Defer to WM-2C with explicit command | Required for assembled QDTWorkingMemory | Deferred register updated |

| QDT-WM-UPG-WM2C-0001 | WM-2C | WMTrace | Trace bus module missing from live source tree | blocker | complete | Implement wm_trace.py | Required for assembly traceability | Source/tests created |
| QDT-WM-UPG-WM2C-0002 | WM-2C | WMTripletState | Triplet state module missing from live source tree | blocker | complete | Implement wm_triplet_state.py | Preserves anchor/direction/phase doctrine | Source/tests created |
| QDT-WM-UPG-WM2C-0003 | WM-2C | Depth adapters | Depth adapters missing from live source tree | blocker | complete | Implement wm_depth_adapters.py | Required before assembly | Source/tests created |
| QDT-WM-UPG-WM2C-0004 | WM-2C | Depth fusion | Depth fusion missing from live source tree | blocker | complete | Implement wm_depth_fusion.py | Required to return [B,T,D] | Source/tests created |
| QDT-WM-UPG-WM2C-0005 | WM-2C | QDTWorkingMemory | Assembled QDT WM missing | blocker | complete | Implement qdt_working_memory.py | Creates usable WM assembly | Source/tests created |
| QDT-WM-UPG-WM2C-PATCH-0001 | WM-2C | QDTWorkingMemoryConfig export | Root export referenced config but wm_config.py was missing | high | complete | Create wm_config.py and patch exports | Prevents config None export | Tests pass |

| QDT-WM-UPG-WM2C-PATCH-0002 | WM-2C | Test runtime stability | Full suite slowed/stalled from PyTorch CPU thread oversubscription in repeated transformer tests | medium | complete | Add tests/conftest.py to set torch threads/inter-op threads to 1 | Makes test suite deterministic and fast | 62 passed in 0.66s |

| QDT-WM-UPG-WM3A-0001 | WM-3A | Retrieval lanes | Multi-lane retrieval modules missing or incomplete | blocker | complete | Implement wm_retrieval_lanes.py | Enables memory-augmented attention | Source/tests created |
| QDT-WM-UPG-WM3A-0002 | WM-3A | Geometry scoring | Candidate scoring/fusion missing | blocker | complete | Implement wm_geometry_scoring.py | Enables lane candidate fusion | Source/tests created |
| QDT-WM-UPG-WM3A-0003 | WM-3A | Geometry linker | Geometry lane routing helper needed but full topology transport deferred | high | complete | Implement wm_geometry_linker.py with explicit deferral notice | Supports route hints without fake topology completion | Source/tests created |
| QDT-WM-UPG-WM3A-0004 | WM-3A | Memory-augmented attention | First MAAE layer missing | blocker | complete | Implement wm_memory_augmented_attention.py | Adds memory context injection | Source/tests created |
| QDT-WM-UPG-WM3A-0005 | WM-3A | PAAMA-X policy lane | Policy lane, write-permission hooks, audit metadata required | blocker | complete | Add policy retrieval lane and paamax metadata | Required for governance chain | Tests pass |
| QDT-WM-UPG-WM3A-0006 | WM-3A | QDTWorkingMemory integration | MAAE must be visible in assembled WM trace | high | complete | Patch qdt_working_memory.py | Integrates attention layer | Tests pass |

| QDT-WM-UPG-WM3A-PATCH-0001 | WM-3A | QDTWorkingMemory MAAE trace integration | MAAE standalone modules existed but read trace did not include memory_augmented_attention stage | blocker | complete | Patch qdt_working_memory.py forward path to call MAAE before depth fusion | Required for QDTWorkingMemory compatibility | Tests rerun and pass |

| QDT-WM-UPG-WM3B-0001 | WM-3B | Evidence attention | Evidence-structured attention missing | blocker | complete | Implement wm_evidence_attention.py | Enables audit/evidence units | Source/tests created |
| QDT-WM-UPG-WM3B-0002 | WM-3B | Trace attention | Trace governance attention missing | high | complete | Implement wm_trace_attention.py | Enables trace-aware routing | Source/tests created |
| QDT-WM-UPG-WM3B-0003 | WM-3B | Counterfactual attention | Counterfactual probe missing | high | complete | Implement wm_counterfactual_attention.py | Enables ablation-style trace metadata | Source/tests created |
| QDT-WM-UPG-WM3B-0004 | WM-3B | Conflict/quarantine attention | Conflict attention missing | blocker | complete | Implement wm_conflict_attention.py | Enables quarantine metadata | Source/tests created |
| QDT-WM-UPG-WM3B-0005 | WM-3B | Novelty/lightbulb attention | Novelty attention missing | high | complete | Implement wm_novelty_attention.py | Enables lightbulb trace metadata | Source/tests created |
| QDT-WM-UPG-WM3B-0006 | WM-3B | Stability attention | Stability-aware attention missing | blocker | complete | Implement wm_stability_attention.py | Enables stability repair guard | Source/tests created |
| QDT-WM-UPG-WM3B-0007 | WM-3B | MAAE integration | Advanced attention modules must be visible in MAAE/QDT trace | blocker | complete | Patch wm_memory_augmented_attention.py and qdt_working_memory.py | Required for integration | Tests pass |

| QDT-WM-UPG-WM4A-0001 | WM-4A | External memory interfaces | LTM/MANN/SPCP query-response contracts missing | blocker | complete | Implement wm_external_memory_interfaces.py | Enables external memory interop | Source/tests created |
| QDT-WM-UPG-WM4A-0002 | WM-4A | LTM cross-attention | LTM WM cross-attention missing | blocker | complete | Implement wm_ltm_cross_attention.py | Enables LTM fusion | Source/tests created |
| QDT-WM-UPG-WM4A-0003 | WM-4A | MANN cross-attention | MANN WM cross-attention and visibility missing | blocker | complete | Implement wm_mann_cross_attention.py | Enables MANN fusion and trace visibility | Source/tests created |
| QDT-WM-UPG-WM4A-0004 | WM-4A | SPCP cross-attention | Procedural memory cross-attention missing | blocker | complete | Implement wm_spcp_cross_attention.py | Enables SPCP fusion | Source/tests created |
| QDT-WM-UPG-WM4A-0005 | WM-4A | Dual fusion controller | LTM/MANN dual-fusion controller missing | blocker | complete | Implement wm_dual_fusion.py | Enables WM + LTM + MANN + SPCP fusion | Source/tests created |
| QDT-WM-UPG-WM4A-0006 | WM-4A | QDTWorkingMemory integration | dual_fusion must appear in assembled WM trace | high | complete | Patch qdt_working_memory.py | Integrates fusion path | Tests pass |

| QDT-WM-UPG-WM4B-0001 | WM-4B | Shared slot registry | Canonical shared slot IDs and registry missing | blocker | complete | Implement wm_shared_slot_registry.py | Enables LTM/MANN shared slots | Source/tests created |
| QDT-WM-UPG-WM4B-0002 | WM-4B | Shared slot store | Store for mirrored metadata and ownership missing | blocker | complete | Implement wm_shared_slot_store.py | Enables mirrored content doctrine | Source/tests created |
| QDT-WM-UPG-WM4B-0003 | WM-4B | External memory shared refs | External memory responses lacked canonical shared refs | high | complete | Patch wm_external_memory_interfaces.py | Enables traceable shared-slot references | Tests pass |
| QDT-WM-UPG-WM4B-0004 | WM-4B | QDTWorkingMemory integration | Shared slot store must be present in QDT WM trace path | high | complete | Patch qdt_working_memory.py | Shared registry visible during read | Tests pass |
| QDT-WM-UPG-WM4B-0005 | WM-4B | Conflict/write metadata | Ownership/conflict/write-permission metadata required | high | complete | Add record/store metadata fields | Supports PAAMA-X gates | Tests pass |

| QDT-WM-UPG-WM4C-0001 | WM-4C | QH storage module | Quantum-holographic depth-coded storage interface missing | blocker | complete | Implement wm_quantum_holographic_storage.py | Enables QH-compatible metadata | Source/tests created |
| QDT-WM-UPG-WM4C-0002 | WM-4C | QH code schema | depth/bank/geometry/triplet/memory/task codes required | blocker | complete | Implement QHCodeSchema and build_qh_code_schema | Preserves QH carryover doctrine | Tests pass |
| QDT-WM-UPG-WM4C-0003 | WM-4C | Shared slot linkage | QH records must link to canonical shared slots | blocker | complete | Patch SharedSlotStore with QH refs | Shared slot/QH integration visible | Tests pass |
| QDT-WM-UPG-WM4C-0004 | WM-4C | Interference checks | QH record interference checks missing | high | complete | Implement QHInterferenceReport | Enables conflict/quarantine metadata | Tests pass |
| QDT-WM-UPG-WM4C-0005 | WM-4C | QDTWorkingMemory QH trace | QH references must appear in WM trace | high | complete | Patch qdt_working_memory.py | QH trace visible on read/process | Tests pass |

| QDT-WM-UPG-WM5A-0001 | WM-5A | System commit gate | Systemwide commit/reject/rollback/quarantine gate missing | blocker | complete | Implement wm_system_commit_gate.py | Enables simultaneous read/write governance | Source/tests created |
| QDT-WM-UPG-WM5A-0002 | WM-5A | PAAMA-X write permission | Systemwide write permission not enforced outside local shadow writes | blocker | complete | Add permission gate in SystemCommitGate | Prevents unauthorized writes | Tests pass |
| QDT-WM-UPG-WM5A-0003 | WM-5A | QH/shared-slot commit integration | Commit path must write shared slot + QH record | blocker | complete | Integrate SharedSlotStore and QuantumHolographicStorage | Storage trace complete | Tests pass |
| QDT-WM-UPG-WM5A-0004 | WM-5A | Rollback/quarantine | Rollback and quarantine paths missing | high | complete | Implement rollback_last and quarantine | Supports recovery and conflict handling | Tests pass |
| QDT-WM-UPG-WM5A-0005 | WM-5A | QDTWorkingMemory write path | QDT write path must use system commit gate | blocker | complete | Patch qdt_working_memory.py | WM write path governed | Tests pass |

| QDT-WM-UPG-WM5A-PATCH-0001 | WM-5A | QDTWorkingMemory constructor order | SystemCommitGate was initialized before shadow_write_buffer existed | blocker | complete | Move SystemCommitGate construction after CurvedShadowWriteBuffer setup | Required for QDTWorkingMemory initialization | Full tests rerun |

| QDT-WM-UPG-WM6A-0001 | WM-6A | Compatibility wrapper | Old EnhancedCurvedMemory-style call compatibility missing | blocker | complete | Implement wm_compatibility_wrapper.py | Enables safe replacement of self.working_memory | Source/tests created |
| QDT-WM-UPG-WM6A-0002 | WM-6A | Cortex migration function | Safe replacement helper missing | blocker | complete | Implement replace_cortex_working_memory | Preserves old reference and installs QDT wrapper | Tests pass |
| QDT-WM-UPG-WM6A-0003 | WM-6A | Minimal cortex shell test | Full EnhancedMnemonicCortex source absent from pack | high | complete | Implement EnhancedMnemonicCortexQDTAdapter for integration proof | Avoids fake patch claims | Tests pass |
| QDT-WM-UPG-WM6A-0004 | WM-6A | Real cortex source patch | Real EnhancedMnemonicCortex source not present in current pack | medium | partially_complete | Provide migration_patch_template; apply real patch when source is supplied | Required for production source tree | Template emitted |
| QDT-WM-UPG-WM6A-0005 | WM-6A | Trace routing | Cortex-level read/process/write trace availability required | high | complete | Add wrapper trace and adapter last_trace | Trace tested | Tests pass |

| QDT-WM-UPG-WM7A-0001 | WM-7A | API surface audit | Final API audit required before release | high | complete | Generate AST-based API surface audit | Supports release review | docs/qdt_wm_maae/100_wm7a_api_surface_audit.md |
| QDT-WM-UPG-WM7A-0002 | WM-7A | Dependency map | Module dependency map required before release | medium | complete | Generate AST-based dependency map | Supports integration review | docs/qdt_wm_maae/101_wm7a_module_dependency_map.md |
| QDT-WM-UPG-WM7A-0003 | WM-7A | Benchmark harness | Release smoke benchmarks missing | high | complete | Add benchmarks/benchmark_qdt_wm_maae.py | Tests runtime smoke behavior | Benchmark results captured |
| QDT-WM-UPG-WM7A-0004 | WM-7A | Release manifest | Manifest required for handoff | high | complete | Generate release manifest | Lists source/test/doc/benchmark files | release/release_manifest.json |
| QDT-WM-UPG-WM7A-0005 | WM-7A | Production readiness | Production caveats must be explicit | high | complete | Generate readiness document | Prevents fake-done production claim | docs/qdt_wm_maae/106_wm7a_production_integration_readiness.md |
| QDT-WM-UPG-WM7A-0006 | WM-7A | Real cortex integration | Real EnhancedMnemonicCortex source absent | medium | partially_complete | Keep deferred item and migration template | Needs real source tree | Deferred register updated |
