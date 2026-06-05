# REASON-1C MANN Depth Integration Design

## Source and token budget

```json
{
  "source_info": {
    "latest_development_pack_exists": true,
    "reason1b_pack_exists": true,
    "wm_pack_exists": true,
    "active_source_pack": "/mnt/data/mnemonic_reasoning_latest_development_integration_pack.zip",
    "active_source_sha256": "415611feeb252100f49141614e6e627567ee228b27c1134591b00ce78e2a319f",
    "reason1b_sha256": "84f5ab3e88a00da50e6720a014c2e071a4d8de1cabb4c7f80fc4295254395140",
    "wm_pack_sha256": "1302773410ceb42172f9fc7b1b26411ff4ed3aad8f126c25d6c6f5de58f90dac",
    "depth_indexed_slot_lattice_exists": true,
    "wm_depth_controller_exists": true,
    "qdt_working_memory_exists": true,
    "mann_related_files_found": [
      "mnemonic_cortex/working_memory/wm_mann_cross_attention.py"
    ]
  },
  "token_budget": {
    "target_scope": "REASON-1C MANN depth SlotKV bank + MANNDepthAdapter, hop traces, shadow write proposals, non-destructive integration hook, tests, docs, package, and print sequence.",
    "minimum_complete_version_tokens_est": 11000,
    "deep_implementation_version_tokens_est": 30000,
    "generated_content_total_tokens_est": 70000,
    "max_safe_response_budget_tokens_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/package completed in one execution; printout split into REASON-1C-PRINT-P1/P2/P3 at file boundaries.",
    "estimated_file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 5,
      "updated_tracker": 1,
      "release_manifest": 1,
      "benchmark_count": 0
    },
    "selected_split_scope": "Full REASON-1C implementation + package; source printout begins in final response.",
    "clean_split_points": [
      "source-integrity audit",
      "MANNSlotKVDepthBank",
      "MANNDepthAdapter",
      "non-destructive MANN integration helper",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "LTM depth adapter integration (REASON-1D)",
      "SPCP procedural bank integration",
      "actual MANN runtime destructive replacement",
      "permanent memory consolidation",
      "model weight mutation",
      "optimizer mutation",
      "direct shared physical tensor storage with LTM",
      "fake quantum hardware backend",
      "real external memory-store activation"
    ]
  }
}
```

## Design

REASON-1C adds a non-destructive MANN depth integration layer. The implementation creates:

- `MANNSlotKVDepthBank`: a slots × 8 key/value bank backed by `DepthIndexedSlotLattice`.
- `MANNDepthAdapter`: a hop-oriented adapter for MANN reasoning reads and shadow write proposals.
- `mann_depth_integration.py`: optional install helper for shell objects.

The bank uses keys `[S,8,K]` and values `[S,8,V]`. Disabled mode is inert/pass-through. Enabled mode returns finite depth-read summaries and emits hop traces.

Hop write proposal routing:

- Z4 reasoning_transform
- Z5 hop_history / temporal_episode
- Z6 candidate_hypothesis
- Z7 scratch / volatile_trace

MANN/LTM sharing remains canonical-ID metadata only. Physical tensor sharing is prohibited.
