# REASON-3D Release Candidate Hardening Design

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
      "REASON-3D-PRINT-P1": {
        "estimated_tokens": 5957,
        "budget_enforced": 7357
      },
      "REASON-3D-PRINT-P1B": {
        "estimated_tokens": 4791,
        "budget_enforced": 5991
      },
      "REASON-3D-PRINT-P2": {
        "estimated_tokens": 1245,
        "budget_enforced": 2445
      },
      "REASON-3D-PRINT-P3": {
        "estimated_tokens": 15500,
        "budget_enforced": 17000
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-3D release candidate hardening, API freeze, regression closure, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 24000,
    "deep_implementation_version_tokens_est": 64000,
    "generated_content_total_tokens_est": 43493,
    "binding_split_decision": "Implementation/tests/package completed in one execution after corrective patch; printout split into REASON-3D-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-3D implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "release candidate",
      "API freeze",
      "regression closure",
      "regression matrix/init patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-4A optional persistence adapter design",
      "automatic persistence writes",
      "permanent memory-store commit execution",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/reasoning_release_candidate.py": 2660,
    "mnemonic_cortex/reasoning_depth/reasoning_api_freeze.py": 1603,
    "mnemonic_cortex/reasoning_depth/reasoning_regression_closure.py": 1694,
    "mnemonic_cortex/reasoning_depth/reasoning_regression_matrix.py": 1126,
    "mnemonic_cortex/reasoning_depth/__init__.py": 3665,
    "tests/test_reason3d_release_candidate_report.py": 205,
    "tests/test_reason3d_api_freeze.py": 195,
    "tests/test_reason3d_regression_closure.py": 172,
    "tests/test_reason3d_no_mutation_and_disabled_defaults.py": 217,
    "tests/test_reason3d_contracts.py": 180,
    "tests/test_reason3d_reason3c_compatibility.py": 162,
    "tests/test_reason3d_api_freeze_blocks_breaking_changes.py": 114
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason3c_planner_quality_pack.zip",
  "primary_source_sha256": "cf901d2a4c944d365be9a4087d606785d5e5558c0c546e3f2d1f424716ec50f5",
  "reason3c_pack_exists": true,
  "latest_pack_exists": true,
  "reasoning_controller_api_exists": true,
  "planner_quality_hardening_exists": true,
  "controller_planner_integration_exists": true,
  "strategy_graph_persistence_readiness_exists": true,
  "regression_matrix_exists": true,
  "public_init_exists": true
}
```

REASON-3D adds a release-candidate report, public API freeze metadata, and regression closure reporting. It is not a production-complete claim and performs no permanent writes.
