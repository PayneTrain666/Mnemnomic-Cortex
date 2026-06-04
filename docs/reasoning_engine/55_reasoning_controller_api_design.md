# REASON-2D Reasoning Controller API Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 2600,
        "budget_enforced": 4100
      },
      "REASON-2D-PRINT-P1": {
        "estimated_tokens": 5573,
        "budget_enforced": 6773
      },
      "REASON-2D-PRINT-P1B": {
        "estimated_tokens": 2173,
        "budget_enforced": 3073
      },
      "REASON-2D-PRINT-P2": {
        "estimated_tokens": 1711,
        "budget_enforced": 2911
      },
      "REASON-2D-PRINT-P3": {
        "estimated_tokens": 14800,
        "budget_enforced": 16300
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-2D controller API hardening, release audit, regression matrix, public import stabilization, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 18000,
    "deep_implementation_version_tokens_est": 50000,
    "generated_content_total_tokens_est": 35457,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-2D-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 8,
      "new_docs": 6,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-2D implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "API wrapper",
      "release audit",
      "regression matrix",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-3A advanced reasoning strategy graph",
      "permanent memory-store commit execution",
      "production persistence adapters",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/reasoning_controller_api.py": 2551,
    "mnemonic_cortex/reasoning_depth/reasoning_release_audit.py": 1923,
    "mnemonic_cortex/reasoning_depth/reasoning_regression_matrix.py": 1099,
    "mnemonic_cortex/reasoning_depth/__init__.py": 2173,
    "tests/test_reason2d_config_serialization.py": 201,
    "tests/test_reason2d_controller_api_disabled.py": 196,
    "tests/test_reason2d_controller_api_enabled.py": 272,
    "tests/test_reason2d_no_mutation_and_defaults.py": 232,
    "tests/test_reason2d_reason2c_compatibility.py": 303,
    "tests/test_reason2d_regression_matrix.py": 145,
    "tests/test_reason2d_release_audit.py": 182,
    "tests/test_reason2d_trace_schema_stability.py": 180
  }
}
```

## Design

REASON-2D hardens the public API boundary around the REASON-2C controller. It adds:

1. `ReasoningControllerAPI` for safe construction and execution.
2. `ReasoningReleaseAudit` for non-mutating release readiness checks.
3. `ReasoningRegressionMatrix` for stage coverage and deferred-work visibility.

Patch `REASON-2D-PATCH-0001` clamps internal enabled controller slot count to at least 8 while preserving the public serialized API config. This prevents small smoke configs from violating underlying MANN SlotKV read-top-k constraints.

All optional policy/evidence/counterfactual/conflict features remain opt-in. Permanent writes remain forbidden through the public API.
