# REASON-4A Persistence Adapter Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 3100,
        "budget_enforced": 4700
      },
      "REASON-4A-PRINT-P1": {
        "estimated_tokens": 6875,
        "budget_enforced": 8275
      },
      "REASON-4A-PRINT-P1B": {
        "estimated_tokens": 4066,
        "budget_enforced": 5266
      },
      "REASON-4A-PRINT-P2": {
        "estimated_tokens": 1654,
        "budget_enforced": 2854
      },
      "REASON-4A-PRINT-P3": {
        "estimated_tokens": 16500,
        "budget_enforced": 18200
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-4A optional persistence adapter design, explicit commit interfaces, store-safety contracts, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 26000,
    "deep_implementation_version_tokens_est": 68000,
    "generated_content_total_tokens_est": 46595,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-4A-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-4A implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "persistence adapter",
      "commit interface",
      "store-safety contracts",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-4B real store backend adapters",
      "automatic persistence writes",
      "permanent memory-store commit execution",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/reasoning_persistence_adapter.py": 2336,
    "mnemonic_cortex/reasoning_depth/reasoning_commit_interface.py": 2683,
    "mnemonic_cortex/reasoning_depth/reasoning_store_safety_contracts.py": 1856,
    "mnemonic_cortex/reasoning_depth/__init__.py": 4066,
    "tests/test_reason4a_persistence_adapter_disabled.py": 191,
    "tests/test_reason4a_persistence_payload_json_safe.py": 233,
    "tests/test_reason4a_commit_interface_rejects_missing_permission.py": 262,
    "tests/test_reason4a_commit_interface_accepts_explicit_intent_metadata_only.py": 263,
    "tests/test_reason4a_store_safety_contracts.py": 208,
    "tests/test_reason4a_no_mutation_by_default.py": 237,
    "tests/test_reason4a_reason3d_compatibility.py": 260
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason3d_release_candidate_pack.zip",
  "primary_source_sha256": "ef42f2e5cdd93493596eebdaff394b8c7dfc9780a00088ca97be732e2bd548ea",
  "reason3d_pack_exists": true,
  "latest_pack_exists": true,
  "release_candidate_exists": true,
  "api_freeze_exists": true,
  "regression_closure_exists": true,
  "public_init_exists": true
}
```

REASON-4A designs metadata-only persistence payloads, explicit commit-intent decisions, and store-safety contracts. It performs no real store writes and keeps automatic persistence disabled.
