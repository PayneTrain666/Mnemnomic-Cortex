# REASON-3C Planner Quality Hardening Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 3000,
        "budget_enforced": 4600
      },
      "REASON-3C-PRINT-P1": {
        "estimated_tokens": 8228,
        "budget_enforced": 9628
      },
      "REASON-3C-PRINT-P1B": {
        "estimated_tokens": 6321,
        "budget_enforced": 7521
      },
      "REASON-3C-PRINT-P2": {
        "estimated_tokens": 1732,
        "budget_enforced": 2932
      },
      "REASON-3C-PRINT-P3": {
        "estimated_tokens": 15000,
        "budget_enforced": 16500
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-3C planner quality hardening, controller integration expansion, strategy-graph persistence readiness, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 24000,
    "deep_implementation_version_tokens_est": 62000,
    "generated_content_total_tokens_est": 46281,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-3C-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-3C implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "planner quality hardening",
      "controller planner integration",
      "strategy graph persistence readiness",
      "api/init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-3D release candidate hardening",
      "automatic persistence writes",
      "automatic remediation patch application",
      "permanent memory-store commit execution",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/planner_quality_hardening.py": 3819,
    "mnemonic_cortex/reasoning_depth/controller_planner_integration.py": 2582,
    "mnemonic_cortex/reasoning_depth/strategy_graph_persistence_readiness.py": 1827,
    "mnemonic_cortex/reasoning_depth/reasoning_controller_api.py": 2988,
    "mnemonic_cortex/reasoning_depth/__init__.py": 3333,
    "tests/test_reason3c_planner_quality_hardening.py": 361,
    "tests/test_reason3c_controller_planner_integration_disabled.py": 195,
    "tests/test_reason3c_controller_planner_integration_enabled.py": 222,
    "tests/test_reason3c_strategy_graph_persistence_readiness.py": 282,
    "tests/test_reason3c_api_integration_opt_in.py": 232,
    "tests/test_reason3c_no_mutation_safety.py": 184,
    "tests/test_reason3c_reason3b_compatibility.py": 256
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason3b_planner_evaluation_pack.zip",
  "primary_source_sha256": "592bd184b2b257c1467804d36cc945f37091f65e6dff571e20e41f52a0bfa133",
  "reason3b_pack_exists": true,
  "latest_pack_exists": true,
  "planner_evaluation_exists": true,
  "planner_failure_classifier_exists": true,
  "planner_remediation_guidance_exists": true,
  "reasoning_controller_api_exists": true,
  "strategy_graph_exists": true,
  "public_init_exists": true
}
```

REASON-3C adds recommendation-only planner hardening, explicit opt-in controller/planner integration, and metadata-only strategy graph persistence readiness. All features are disabled or inert by default and perform no permanent memory-store mutation.
