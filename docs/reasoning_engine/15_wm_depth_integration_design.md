# REASON-1B WM Depth Integration Design

## Source and token budget

```json
{
  "source_info": {
    "reason1a_pack_exists": true,
    "context_pack_exists": true,
    "wm_pack_exists": true,
    "active_source_pack": "/mnt/data/mnemonic_reasoning_reason1a_depth_lattice_pack.zip",
    "reason1a_sha256": "a614a7289913c2636f7e03f3b5bc4ed9b1387d06c0c227dcc1fb553e0cbe99b4",
    "context_pack_sha256": "078ffc7874e4bca1c996e1a3c5569f26e7be712901dbbbbf3def7dfb7393e02f",
    "wm_pack_sha256": "1302773410ceb42172f9fc7b1b26411ff4ed3aad8f126c25d6c6f5de58f90dac",
    "qdt_working_memory_source_exists": true,
    "context_compression_source_exists": true,
    "wm_context_mount_source_exists": true
  },
  "token_budget": {
    "target_scope": "REASON-1B optional WM depth adapter/controller integration for DepthIndexedSlotLattice, context-compression candidate routing, QDT compatibility, tests, docs, package, and print sequence.",
    "minimum_complete_version_tokens_est": 9500,
    "deep_implementation_version_tokens_est": 26000,
    "generated_content_total_tokens_est": 60000,
    "max_safe_response_tokens_for_print_p1_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/package completed in one execution; printout split into REASON-1B-PRINT-P1/P2/P3 at file boundaries.",
    "estimated_file_module_count": {
      "new_source_files": 2,
      "patched_source_files": 2,
      "new_tests": 6,
      "new_docs": 5,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "clean_split_points": [
      "source pack extraction and integrity audit",
      "wm_depth_adapter",
      "wm_depth_controller",
      "optional exports and integration helpers",
      "context compression candidate routing",
      "tests",
      "docs/tracker/ship-check",
      "print sequence"
    ],
    "selected_split_scope": "Full REASON-1B implementation + package; source printout begins in final response.",
    "explicit_out_of_scope": [
      "MANN depth adapter integration (REASON-1C)",
      "LTM depth adapter integration (REASON-1D)",
      "permanent memory consolidation",
      "model weight mutation",
      "optimizer mutation",
      "destructive replacement of QDTWorkingMemory",
      "fake quantum hardware backend",
      "real external memory-store activation"
    ]
  }
}
```

## Design

REASON-1B adds an optional `WMDepthAdapter` and `WMDepthController` on top of the REASON-1A `DepthIndexedSlotLattice`.

Default behavior is disabled/pass-through. When enabled, WM state `[B,D]` or `[B,T,D]` can be read through a WM depth lattice and return a depth summary `[B,V]`.

Context-compression candidates are routed as shadow write proposals into:

- Z3 contextual binding
- Z5 temporal episode
- Z7 volatile trace

No QDTWorkingMemory runtime is destructively replaced. The working-memory package receives only an optional integration helper.
