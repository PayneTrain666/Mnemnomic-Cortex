# REASON-2C Evidence / Counterfactual Design

## Dynamic token budget policy resolution

```json
{
  "global_policy_resolution": {
    "fixed_18000_rule_removed": true,
    "rule": "Precompute estimated token cost from selected source/test/doc content before each response; set max safe response budget per response/split from actual selected content, not as a universal constant.",
    "response_split_budgets": {
      "implementation_status_response": {
        "estimated_tokens": 2500,
        "budget_enforced": 4000
      },
      "REASON-2C-PRINT-P1": {
        "estimated_tokens": 5504,
        "budget_enforced": 6704
      },
      "REASON-2C-PRINT-P1B": {
        "estimated_tokens": 6774,
        "budget_enforced": 7974
      },
      "REASON-2C-PRINT-P2": {
        "estimated_tokens": 2001,
        "budget_enforced": 3201
      },
      "REASON-2C-PRINT-P3": {
        "estimated_tokens": 15000,
        "budget_enforced": 16500
      }
    }
  },
  "stage_budget": {
    "target_scope": "REASON-2C evidence reasoning, counterfactual probes, conflict-aware consolidation, controller integration patch, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 17000,
    "deep_implementation_version_tokens_est": 52000,
    "generated_content_total_tokens_est": 42279,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-2C-PRINT-P1/P1B/P2/P3 using dynamic precomputed budgets.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 6,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-2C implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "dynamic token budget policy",
      "evidence pass",
      "counterfactual probe",
      "conflict evaluator",
      "controller integration",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "controller API hardening/release readiness (REASON-2D)",
      "permanent memory-store commit execution",
      "production external adapter activation",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "unbounded counterfactual search",
      "real ablation execution",
      "fake production-complete claim"
    ]
  },
  "section_budgets": [
    {
      "section": "0 source-of-truth and source-integrity check",
      "estimated_tokens": 900,
      "budget_enforced": 1200
    },
    {
      "section": "1 dynamic token budget precomputation policy",
      "estimated_tokens": 1300,
      "budget_enforced": 1700
    },
    {
      "section": "2 evidence_reasoning_pass.py",
      "estimated_tokens": 2054,
      "budget_enforced": 2654
    },
    {
      "section": "3 counterfactual_reasoning_probe.py",
      "estimated_tokens": 1647,
      "budget_enforced": 2247
    },
    {
      "section": "4 conflict_aware_consolidation.py",
      "estimated_tokens": 1803,
      "budget_enforced": 2403
    },
    {
      "section": "5 ReasoningController + init patch",
      "estimated_tokens": 6774,
      "budget_enforced": 7774
    },
    {
      "section": "6 tests",
      "estimated_tokens": 2001,
      "budget_enforced": 2901
    },
    {
      "section": "7 docs/tracker/manifest",
      "estimated_tokens": 12500,
      "budget_enforced": 14000
    },
    {
      "section": "8 package and print splits",
      "estimated_tokens": 1800,
      "budget_enforced": 2400
    }
  ],
  "file_token_estimates": {
    "mnemonic_cortex/reasoning_depth/evidence_reasoning_pass.py": 2054,
    "mnemonic_cortex/reasoning_depth/counterfactual_reasoning_probe.py": 1647,
    "mnemonic_cortex/reasoning_depth/conflict_aware_consolidation.py": 1803,
    "mnemonic_cortex/reasoning_depth/reasoning_controller.py": 4908,
    "mnemonic_cortex/reasoning_depth/__init__.py": 1866,
    "tests/test_reason2c_evidence_reasoning_pass.py": 264,
    "tests/test_reason2c_counterfactual_probe.py": 234,
    "tests/test_reason2c_conflict_aware_consolidation.py": 333,
    "tests/test_reason2c_controller_optional_integration.py": 445,
    "tests/test_reason2c_disabled_default_compatibility.py": 213,
    "tests/test_reason2c_reason2b_compatibility.py": 272,
    "tests/test_reason2c_trace_serialization_no_mutation.py": 240
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason2b_policy_router_pack.zip",
  "primary_source_sha256": "a3b7590f4a86ef60724a52005741f1d520318d86a65f2a709c6ad20bf80c637b",
  "reason2b_pack_exists": true,
  "latest_pack_exists": true,
  "reasoning_controller_exists": true,
  "reason2b_modules_present": true
}
```

## Design

REASON-2C adds three optional controller passes:

1. `EvidenceReasoningPass` creates bounded evidence units from content.
2. `CounterfactualReasoningProbe` performs metadata-only evidence ablation estimates without real ablation execution.
3. `ConflictAwareConsolidationEvaluator` adjusts confidence/disagreement and recommends quarantine metadata before consolidation gate evaluation.

All three are disabled by default and non-mutating.
