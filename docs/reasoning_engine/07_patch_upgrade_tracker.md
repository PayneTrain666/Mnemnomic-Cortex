# Reasoning Engine Patch / Upgrade Tracker

| tracker_id | stage | affected files/modules/docs/tests | reason | severity | status | required action | downstream impact | completion evidence |
|---|---|---|---|---|---|---|---|---|
| REASON-1A-0001 | REASON-1A | mnemonic_cortex/reasoning_depth/depth_indexed_slot_lattice.py | DepthIndexedSlotLattice missing | blocker | complete | Implement slots × 8 depth lattice | Enables WM/MANN/LTM adapters | Source + tests generated |
| REASON-1A-0002 | REASON-1A | mnemonic_cortex/reasoning_depth/shared_depth_slot_registry.py | SharedDepthSlotRegistry missing | high | complete | Implement canonical ID registry without shared physical tensors | Enables safe WM/MANN/LTM mirroring | Source + tests generated |
| REASON-1A-0003 | REASON-1B | WM depth adapter | WM integration intentionally deferred | high | incomplete | Implement optional WM adapter | Required for active WM use | Deferred to REASON-1B |
| REASON-1A-0004 | REASON-1C | MANN depth adapter | MANN integration intentionally deferred | high | incomplete | Implement MANN SlotKV depth adapter | Required for Hop Engine | Deferred to REASON-1C |
| REASON-1A-0005 | REASON-1D | LTM depth adapter | LTM integration intentionally deferred | high | incomplete | Implement LTM shared-depth adapter | Required for LTM/MANN shared canonical slots | Deferred to REASON-1D |
| REASON-1A-0006 | REASON-1A | printout sequence | Full generated content too large for one response | medium | partially_complete | Split printout into P1/P2/P3 | Prevents compressed output | PRINT-P1 starts after implementation |

| REASON-1B-0001 | REASON-1B | mnemonic_cortex/reasoning_depth/wm_depth_adapter.py | WM depth adapter missing | high | complete | Implement optional disabled-by-default WM depth adapter | Enables active WM depth read/proposal routing | Source + tests generated |
| REASON-1B-0002 | REASON-1B | mnemonic_cortex/reasoning_depth/wm_depth_controller.py | WM depth controller missing | high | complete | Implement controller wrapper for future QDT integration | Enables non-destructive controller attachment | Source + tests generated |
| REASON-1B-0003 | REASON-1B | mnemonic_cortex/working_memory/wm_depth_integration.py | Optional WM integration hook missing | medium | complete | Implement attach helper without replacing QDTWorkingMemory | Enables later cortex/WM shell wiring | Source + tests generated |
| REASON-1B-0004 | REASON-1B | context compression candidate routing | Context candidates not routed to Z3/Z5/Z7 | high | complete | Add shadow proposals for contextual/episode/volatile depths | Enables project/chat episodic proposals | Tests pass |
| REASON-1B-0005 | REASON-1C | MANN depth adapter | MANN integration deferred | high | incomplete | Implement MANN SlotKV depth adapter | Required for Hop Engine | Deferred to REASON-1C |
| REASON-1B-0006 | REASON-1B | printout sequence | Full generated content too large for one response | medium | partially_complete | Split printout into P1/P2/P3 | Prevents compressed output | PRINT-P1 starts after implementation |

| REASON-1C-0001 | REASON-1C | mnemonic_cortex/reasoning_depth/mann_slotkv_depth_bank.py | MANN SlotKV depth bank missing | high | complete | Implement MANN keys [S,8,K] and values [S,8,V] bank | Enables MANN depth read/write proposals | Source + tests generated |
| REASON-1C-0002 | REASON-1C | mnemonic_cortex/reasoning_depth/mann_depth_adapter.py | MANN depth adapter missing | high | complete | Implement optional MANNDepthAdapter | Enables hop-oriented depth routing | Source + tests generated |
| REASON-1C-0003 | REASON-1C | MANN hop trace | Hop-oriented MANN depth trace missing | high | complete | Emit hop_id, slots, depths, entropy, support/confidence/disagreement | Required for Hop Engine | Tests pass |
| REASON-1C-0004 | REASON-1C | MANN shadow write proposals | MANN shadow write proposal routing missing | high | complete | Add Z4/Z5/Z6/Z7 proposals | Enables safe reasoning scratchpad writes | Tests pass |
| REASON-1C-0005 | REASON-1D | LTM depth adapter | LTM integration deferred | high | incomplete | Implement LTM depth adapter and registry strengthening | Required for LTM/MANN canonical sharing | Deferred to REASON-1D |
| REASON-1C-0006 | REASON-1C | printout sequence | Full generated content too large for one response | medium | partially_complete | Split printout into P1/P2/P3 | Prevents compressed output | PRINT-P1 starts after implementation |

| REASON-1D-PATCH-0001 | REASON-1D | packaging/source selection | Initial REASON-1D attempt used mutable latest pack after drift and lost prior tests | high | complete | Rebuild from fixed REASON-1C source pack before updating latest integration ZIP | Restores lineage and test coverage | Corrected package built from REASON-1C pack |
| REASON-1D-0001 | REASON-1D | mnemonic_cortex/reasoning_depth/ltm_depth_banks.py | LTM depth banks missing | high | complete | Implement HG/CGMN/spatial/procedural depth banks | Enables LTM depth storage/readiness | Source + tests generated |
| REASON-1D-0002 | REASON-1D | mnemonic_cortex/reasoning_depth/ltm_depth_adapter.py | LTM depth adapter missing | high | complete | Implement optional LTMDepthAdapter | Enables LTM depth reads and shadow consolidation | Source + tests generated |
| REASON-1D-0003 | REASON-1D | mnemonic_cortex/reasoning_depth/shared_depth_slot_registry.py | Registry provenance/consolidation fields missing | high | complete | Add source_stage/source_pack/consolidation_status/lineage fields | Enables WM/MANN/LTM provenance without shared tensors | Tests pass |
| REASON-1D-0004 | REASON-1D | shadow consolidation proposal routing | Shadow consolidation proposal routing missing | high | complete | Implement LTMDepthAdapter.propose_consolidation | Enables gated LTM consolidation proposals | Tests pass |
| REASON-1D-0005 | REASON-1E | benchmark/capacity validation | Benchmark/capacity validation deferred | medium | incomplete | Implement depth-lattice capacity/readiness benchmarks | Required before controller/orchestrator integration | Deferred to REASON-1E |
| REASON-1D-0006 | REASON-1D | printout sequence | Full generated content too large for one response | medium | partially_complete | Split printout into P1/P1B/P2/P3 | Prevents compressed output | PRINT-P1 starts after implementation |
