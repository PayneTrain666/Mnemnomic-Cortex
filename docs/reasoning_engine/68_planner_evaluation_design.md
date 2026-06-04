# REASON-3B Planner Evaluation Design

## Dynamic token budget policy

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set split budget from selected content plus margin.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 2800,
        "budget_enforced": 4300
      },
      "REASON-3B-PRINT-P1": {
        "estimated_tokens": 9279,
        "budget_enforced": 10679
      },
      "REASON-3B-PRINT-P1B": {
        "estimated_tokens": 2937,
        "budget_enforced": 3937
      },
      "REASON-3B-PRINT-P2": {
        "estimated_tokens": 1791,
        "budget_enforced": 2991
      },
      "REASON-3B-PRINT-P3": {
        "estimated_tokens": 15000,
        "budget_enforced": 16500
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-3B planner evaluation, failure classification, remediation guidance, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 22000,
    "deep_implementation_version_tokens_est": 58000,
    "generated_content_total_tokens_est": 44007,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-3B-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 8,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-3B implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "planner evaluation",
      "failure classifier",
      "remediation guidance",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-3C planner quality hardening",
      "automatic remediation patch application",
      "permanent memory-store commit execution",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/planner_evaluation.py": 3373,
    "mnemonic_cortex/reasoning_depth/planner_failure_classifier.py": 3299,
    "mnemonic_cortex/reasoning_depth/planner_remediation_guidance.py": 2607,
    "mnemonic_cortex/reasoning_depth/__init__.py": 2937,
    "tests/test_reason3b_planner_evaluation_disabled.py": 158,
    "tests/test_reason3b_planner_evaluation_enabled.py": 280,
    "tests/test_reason3b_failure_classifier.py": 227,
    "tests/test_reason3b_failure_classifier_bounds.py": 275,
    "tests/test_reason3b_remediation_guidance.py": 287,
    "tests/test_reason3b_remediation_no_auto_patch.py": 123,
    "tests/test_reason3b_no_mutation_and_serialization.py": 190,
    "tests/test_reason3b_reason3a_compatibility.py": 251
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason3a_strategy_graph_planner_pack.zip",
  "primary_source_sha256": "4a60da7846e0f86ce2dcdd6cb868bb78a4cd3631ed846c7540615854ea9b912d",
  "reason3a_pack_exists": true,
  "latest_pack_exists": true,
  "reasoning_strategy_graph_exists": true,
  "multi_pass_thought_planner_exists": true,
  "evidence_guided_route_expander_exists": true,
  "reasoning_controller_api_exists": true,
  "public_init_exists": true
}
```

REASON-3B adds `PlannerEvaluator`, a disabled-by-default, non-mutating evaluator for REASON-3A `ThoughtPlanReport` payloads. It scores pass count, route count, confidence, disagreement, evidence support, unsupported routes, conflict-prone routes, boundedness, and no-mutation safety.
