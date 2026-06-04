# FUTURE-BACKEND-AUTHORIZATION Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 3400,
        "budget_enforced": 5000
      },
      "FUTURE-BACKEND-AUTH-PRINT-P1": {
        "estimated_tokens": 5778,
        "budget_enforced": 7278
      },
      "FUTURE-BACKEND-AUTH-PRINT-P1B": {
        "estimated_tokens": 5293,
        "budget_enforced": 6493
      },
      "FUTURE-BACKEND-AUTH-PRINT-P2": {
        "estimated_tokens": 1384,
        "budget_enforced": 2584
      },
      "FUTURE-BACKEND-AUTH-PRINT-P3": {
        "estimated_tokens": 18500,
        "budget_enforced": 20200
      }
    }
  },
  "stage_budget": {
    "target_scope": "FUTURE-BACKEND-AUTHORIZATION planning-only backend authorization, threat model, implementation plan, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 28000,
    "deep_implementation_version_tokens_est": 74000,
    "generated_content_total_tokens_est": 50455,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full planning-only authorization stage and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "authorization gate",
      "threat model",
      "implementation plan",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "write-capable backend code",
      "real persistence backend implementation",
      "credential loading",
      "schema migration execution",
      "automatic persistence writes",
      "permanent memory-store commit execution",
      "real external backend connections",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/backend_authorization.py": 1939,
    "mnemonic_cortex/reasoning_depth/backend_threat_model.py": 1693,
    "mnemonic_cortex/reasoning_depth/backend_implementation_plan.py": 2146,
    "mnemonic_cortex/reasoning_depth/__init__.py": 5293,
    "tests/test_future_backend_authorization_gate.py": 216,
    "tests/test_future_backend_authorization_disabled.py": 150,
    "tests/test_future_backend_threat_model.py": 168,
    "tests/test_future_backend_implementation_plan.py": 198,
    "tests/test_future_backend_no_write_capable_code_guards.py": 189,
    "tests/test_future_backend_reason4c_compatibility.py": 240,
    "tests/test_future_backend_json_safety.py": 223
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason4c_persistence_closure_pack.zip",
  "primary_source_sha256": "4d6dad24ed060c575e44eeacc64f0448110951fa93c23433ee0c6920952613fd",
  "reason4c_pack_exists": true,
  "latest_pack_exists": true,
  "persistence_line_closure_exists": true,
  "integration_index_exists": true,
  "final_safety_audit_exists": true,
  "public_init_exists": true
}
```

This stage authorizes planning only. It does not authorize real persistence writes, credential loading, schema migration execution, or write-capable backend code.
