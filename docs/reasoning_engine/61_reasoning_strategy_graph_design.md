# REASON-3A Reasoning Strategy Graph Design

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
      "REASON-3A-PRINT-P1": {
        "estimated_tokens": 8492,
        "budget_enforced": 9892
      },
      "REASON-3A-PRINT-P1B": {
        "estimated_tokens": 5304,
        "budget_enforced": 6304
      },
      "REASON-3A-PRINT-P2": {
        "estimated_tokens": 1924,
        "budget_enforced": 3124
      },
      "REASON-3A-PRINT-P3": {
        "estimated_tokens": 15000,
        "budget_enforced": 16500
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-3A strategy graph, multi-pass planner, evidence-guided route expansion, optional controller API integration, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 21000,
    "deep_implementation_version_tokens_est": 56000,
    "generated_content_total_tokens_est": 44220,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-3A-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 8,
      "new_docs": 7,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-3A implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "strategy graph",
      "route expander",
      "multi-pass planner",
      "controller API opt-in patch",
      "init export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "REASON-3B planner evaluation and failure classification",
      "permanent memory-store commit execution",
      "production persistence adapters",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "fake production-complete claim"
    ]
  },
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/reasoning_strategy_graph.py": 3219,
    "mnemonic_cortex/reasoning_depth/multi_pass_thought_planner.py": 3058,
    "mnemonic_cortex/reasoning_depth/evidence_guided_route_expander.py": 2215,
    "mnemonic_cortex/reasoning_depth/reasoning_controller_api.py": 2752,
    "mnemonic_cortex/reasoning_depth/__init__.py": 2552,
    "tests/test_reason3a_strategy_graph_serialization.py": 238,
    "tests/test_reason3a_strategy_graph_bounds.py": 181,
    "tests/test_reason3a_multi_pass_planner_disabled.py": 174,
    "tests/test_reason3a_multi_pass_planner_enabled.py": 293,
    "tests/test_reason3a_evidence_guided_route_expander.py": 317,
    "tests/test_reason3a_no_mutation_and_trace.py": 193,
    "tests/test_reason3a_controller_api_compatibility.py": 276,
    "tests/test_reason3a_reason2d_compatibility.py": 252
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason2d_controller_api_release_pack.zip",
  "primary_source_sha256": "d39777806263d01377615ee62c068ef688b4a51f0d8b9fcc38526b20fadfd563",
  "reason2d_pack_exists": true,
  "latest_pack_exists": true,
  "reasoning_controller_api_exists": true,
  "reasoning_release_audit_exists": true,
  "reasoning_regression_matrix_exists": true,
  "reasoning_controller_exists": true,
  "evidence_reasoning_pass_exists": true,
  "reasoning_policy_router_exists": true,
  "public_init_exists": true
}
```

## Design

REASON-3A adds a bounded, JSON-safe, metadata-only strategy graph. The graph creates route candidates for multi-pass planning without writing to WM/MANN/LTM stores.

Core objects:
- `ReasoningStrategyGraphConfig`
- `ReasoningStrategyNode`
- `ReasoningStrategyEdge`
- `ReasoningStrategyGraph`

The graph is disabled by default, bounded by max nodes/edges, and exports JSON-safe route candidates.
