# REASON-1A Depth Lattice Design

## Source and token budget

```json
{
  "source_info": {
    "wm_pack_exists": true,
    "context_pack_exists": true,
    "active_source_pack": "/mnt/data/qdt_wm_maae_context_compression_memory_pack.zip",
    "active_source_sha256": "078ffc7874e4bca1c996e1a3c5569f26e7be712901dbbbbf3def7dfb7393e02f"
  },
  "token_budget": {
    "target_scope": "REASON-1A foundational DepthIndexedSlotLattice implementation for slots \u00d7 8 structured depth layers across WM/MANN/LTM future adapters.",
    "minimum_complete_version_tokens_est": 9000,
    "deep_implementation_version_tokens_est": 24000,
    "generated_content_total_tokens_est": 56000,
    "max_safe_response_tokens_for_print_p1_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/package completed in one execution; printout split into REASON-1A-PRINT-P1/P2/P3 at file boundaries.",
    "estimated_file_module_count": {
      "new_source_files": 10,
      "patched_source_files": 0,
      "new_tests": 7,
      "new_docs": 5,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "clean_split_points": [
      "source pack extraction and source-grounding",
      "config/types/trace",
      "entropy/attention/write policy/capacity",
      "DepthIndexedSlotLattice",
      "SharedDepthSlotRegistry",
      "tests",
      "docs/tracker/ship-check",
      "print sequence"
    ],
    "selected_split_scope": "Full REASON-1A implementation + package; source printout begins in final response.",
    "explicit_out_of_scope": [
      "WM adapter integration (REASON-1B)",
      "MANN adapter integration (REASON-1C)",
      "LTM adapter integration (REASON-1D)",
      "SPCP procedural bank integration",
      "Permanent consolidation writes",
      "Model weight mutation",
      "Optimizer mutation",
      "Fake quantum hardware backend",
      "Destructive replacement of QDTWorkingMemory"
    ]
  }
}
```

## Design

REASON-1A implements physical `slots × 8 depth` tensors rather than a flat `slots*8` bank. Slot identity and depth identity remain separate, so one canonical memory object can store multiple structured aspects.

Canonical tensor shapes:

- keys: `[S,8,K]`
- values: `[S,8,V]`
- importance/confidence/usage/age: `[S,8]`

Depth roles are core identity, semantic invariant, structural relation, contextual binding, reasoning transform, temporal episode, experimental hypothesis, and volatile trace.
