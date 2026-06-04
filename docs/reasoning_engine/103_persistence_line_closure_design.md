# REASON-4C Persistence Line Closure Design

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
      "REASON-4C-PRINT-P1": {
        "estimated_tokens": 6056,
        "budget_enforced": 7456
      },
      "REASON-4C-PRINT-P1B": {
        "estimated_tokens": 4883,
        "budget_enforced": 6083
      },
      "REASON-4C-PRINT-P2": {
        "estimated_tokens": 1344,
        "budget_enforced": 2544
      },
      "REASON-4C-PRINT-P3": {
        "estimated_tokens": 18000,
        "budget_enforced": 19700
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-4C persistence line closure, final safety audit, integration index, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 27000,
    "deep_implementation_version_tokens_est": 72000,
    "generated_content_total_tokens_est": 48283,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-4C-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-4C implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "persistence line closure",
      "integration index",
      "final safety audit",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "real persistence backend implementation",
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
    "mnemonic_cortex/reasoning_depth/reasoning_persistence_line_closure.py": 2113,
    "mnemonic_cortex/reasoning_depth/reasoning_integration_index.py": 2173,
    "mnemonic_cortex/reasoning_depth/reasoning_final_safety_audit.py": 1770,
    "mnemonic_cortex/reasoning_depth/__init__.py": 4883,
    "tests/test_reason4c_persistence_line_closure.py": 191,
    "tests/test_reason4c_integration_index.py": 171,
    "tests/test_reason4c_final_safety_audit.py": 177,
    "tests/test_reason4c_disabled_defaults.py": 183,
    "tests/test_reason4c_no_real_write_guards.py": 169,
    "tests/test_reason4c_reason4b_compatibility.py": 274,
    "tests/test_reason4c_final_closure_command_policy.py": 179
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason4b_persistence_backend_pack.zip",
  "primary_source_sha256": "ae63fa5fea8e1014b1a99563a1201b4598dd21f6b82783a43e88ba9126879aed",
  "reason4b_pack_exists": true,
  "latest_pack_exists": true,
  "persistence_backends_exists": true,
  "dry_run_ledger_exists": true,
  "persistence_recovery_exists": true,
  "public_init_exists": true,
  "known_reason4b_doc_name_mismatch": true
}
```

REASON-4C closes the persistence-design line in a metadata-only state. It adds a decision register, integration index, and final safety audit. Real backend implementation is explicitly deferred behind a future authorization command.
