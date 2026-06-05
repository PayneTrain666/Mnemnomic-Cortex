# REASON-1D LTM Depth Integration Design

## Source and token budget

```json
{
  "source_info": {
    "latest_development_pack_rebuilt_after_failed_attempt": true,
    "reason1c_pack_exists": true,
    "wm_pack_exists": true,
    "active_source_pack": "/mnt/data/mnemonic_reasoning_reason1c_mann_depth_integration_pack.zip",
    "active_source_sha256": "1eea848b93ea5a736f8fbec8b9e9a1169d8abe1cc12819a823f6670861313d93",
    "wm_pack_sha256": "1302773410ceb42172f9fc7b1b26411ff4ed3aad8f126c25d6c6f5de58f90dac",
    "depth_indexed_slot_lattice_exists": true,
    "mann_depth_adapter_exists": true,
    "shared_depth_slot_registry_exists": true,
    "reason1c_tests_exist": true,
    "ltm_related_files_found": [
      "mnemonic_cortex/working_memory/wm_ltm_cross_attention.py"
    ]
  },
  "token_budget": {
    "target_scope": "REASON-1D LTM depth banks + LTMDepthAdapter, SharedDepthSlotRegistry provenance/consolidation extension, shadow consolidation proposals, tests, docs, package, and print sequence.",
    "minimum_complete_version_tokens_est": 12500,
    "deep_implementation_version_tokens_est": 34000,
    "generated_content_total_tokens_est": 76000,
    "max_safe_response_budget_tokens_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/package completed in one execution; printout split into REASON-1D-PRINT-P1/P1B/P2/P3 at file boundaries.",
    "estimated_file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 5,
      "updated_tracker": 1,
      "release_manifest": 1,
      "benchmark_count": 0
    },
    "selected_split_scope": "Full REASON-1D implementation + package; source printout begins in final response.",
    "clean_split_points": [
      "source-integrity audit",
      "SharedDepthSlotRegistry extension",
      "LTMDepthBanks",
      "LTMDepthAdapter",
      "non-destructive LTM integration helper",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "depth-lattice benchmarks (REASON-1E)",
      "SPCP procedural runtime implementation",
      "actual LTM runtime destructive replacement",
      "permanent memory consolidation",
      "model weight mutation",
      "optimizer mutation",
      "direct MANN/LTM shared physical tensor storage",
      "fake quantum hardware backend",
      "real external memory-store activation"
    ]
  }
}
```

## Design

REASON-1D adds non-destructive LTM depth banks and a shadow-consolidation adapter on top of the existing slots × 8 depth lattice.

Created LTM banks:

- `hg_episodic`
- `cgmn_semantic`
- `spatial_topological`
- `procedural_spcp`

Each bank uses keys `[S,8,K]` and values `[S,8,V]`. The procedural/SPCP bank is metadata-compatible and safe; it does not claim a real quantum or hardware backend.

The SharedDepthSlotRegistry is extended with provenance and consolidation fields while preserving old API compatibility.
