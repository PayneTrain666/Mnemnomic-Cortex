# REASON-4B Persistence Backend Stubs Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 3200,
        "budget_enforced": 4800
      },
      "REASON-4B-PRINT-P1": {
        "estimated_tokens": 5707,
        "budget_enforced": 7107
      },
      "REASON-4B-PRINT-P1B": {
        "estimated_tokens": 4483,
        "budget_enforced": 5683
      },
      "REASON-4B-PRINT-P2": {
        "estimated_tokens": 1408,
        "budget_enforced": 2608
      },
      "REASON-4B-PRINT-P3": {
        "estimated_tokens": 17500,
        "budget_enforced": 19200
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-4B persistence backend stubs, dry-run commit ledger, recovery semantics, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 27000,
    "deep_implementation_version_tokens_est": 70000,
    "generated_content_total_tokens_est": 47598,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-4B-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-4B implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "backend stubs",
      "dry-run ledger",
      "recovery semantics",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-4C final closure or explicit backend implementation",
      "automatic persistence writes",
      "permanent memory-store commit execution",
      "real external backend connections",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/reasoning_persistence_backends.py": 2097,
    "mnemonic_cortex/reasoning_depth/reasoning_commit_dry_run_ledger.py": 1765,
    "mnemonic_cortex/reasoning_depth/reasoning_persistence_recovery.py": 1845,
    "mnemonic_cortex/reasoning_depth/__init__.py": 4483,
    "tests/test_reason4b_backend_disabled_defaults.py": 177,
    "tests/test_reason4b_backend_dry_run_acceptance.py": 171,
    "tests/test_reason4b_commit_dry_run_ledger.py": 205,
    "tests/test_reason4b_ledger_idempotency.py": 186,
    "tests/test_reason4b_persistence_recovery.py": 237,
    "tests/test_reason4b_no_real_write_guards.py": 165,
    "tests/test_reason4b_reason4a_compatibility.py": 267
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason4a_persistence_adapter_pack.zip",
  "primary_source_sha256": "aaec4c7cf7198050c4d73915688ec0196da153ee90469a85a962a9b23c27bbdb",
  "reason4a_pack_exists": true,
  "latest_pack_exists": true,
  "persistence_adapter_exists": true,
  "commit_interface_exists": true,
  "store_safety_contracts_exists": true,
  "public_init_exists": true
}
```

REASON-4B creates dry-run-only backend stubs, an in-memory dry-run ledger, and metadata-only recovery planning. No file, database, graph, vector-store, or external writes are performed.
