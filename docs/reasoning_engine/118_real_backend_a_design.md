# REAL-BACKEND-IMPLEMENTATION-A Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 3600,
        "budget_enforced": 5200
      },
      "REAL-BACKEND-A-PRINT-P1": {
        "estimated_tokens": 5715,
        "budget_enforced": 7315
      },
      "REAL-BACKEND-A-PRINT-P1B": {
        "estimated_tokens": 5805,
        "budget_enforced": 7005
      },
      "REAL-BACKEND-A-PRINT-P2": {
        "estimated_tokens": 1679,
        "budget_enforced": 2879
      },
      "REAL-BACKEND-A-PRINT-P3": {
        "estimated_tokens": 19500,
        "budget_enforced": 21200
      }
    }
  },
  "stage_budget": {
    "target_scope": "REAL-BACKEND-IMPLEMENTATION-A dry-run backend interface, credential scope, backup/recovery plan, migration dry-run plan, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 32000,
    "deep_implementation_version_tokens_est": 82000,
    "generated_content_total_tokens_est": 55199,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 4,
      "patched_source_files": 1,
      "new_tests": 8,
      "new_docs": 8,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full dry-run-first backend interface stage and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "backend interface protocol",
      "dry-run backend implementation",
      "credential scope model",
      "backup/recovery planner",
      "migration dry-run planner",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "real persistence writes",
      "write permission grant",
      "real credential loading",
      "schema migration execution",
      "real backup/restore execution",
      "external backend connections",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/backend_interface_protocol.py": 1256,
    "mnemonic_cortex/reasoning_depth/dry_run_backend_interface.py": 1085,
    "mnemonic_cortex/reasoning_depth/credential_scope_model.py": 1187,
    "mnemonic_cortex/reasoning_depth/backend_backup_recovery.py": 2187,
    "mnemonic_cortex/reasoning_depth/__init__.py": 5805,
    "tests/test_real_backend_a_interface_disabled.py": 197,
    "tests/test_real_backend_a_dry_run_write.py": 238,
    "tests/test_real_backend_a_idempotency.py": 186,
    "tests/test_real_backend_a_credential_scope.py": 196,
    "tests/test_real_backend_a_backup_recovery.py": 215,
    "tests/test_real_backend_a_migration_dry_run.py": 200,
    "tests/test_real_backend_a_future_auth_compatibility.py": 288,
    "tests/test_real_backend_a_dependency_check.py": 159
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_future_backend_authorization_pack.zip",
  "primary_source_sha256": "4609830d9c838552e2a2bfb449db4efb995860670958f9a4ce8ca8b6c256b654",
  "future_auth_pack_exists": true,
  "latest_pack_exists": true,
  "backend_authorization_exists": true,
  "backend_threat_model_exists": true,
  "backend_implementation_plan_exists": true,
  "public_init_exists": true
}
```

This stage creates dry-run-first backend interface code only. It does not authorize or implement real persistence writes, credential loading, schema migration execution, external backend connections, or permanent memory-store mutation.
