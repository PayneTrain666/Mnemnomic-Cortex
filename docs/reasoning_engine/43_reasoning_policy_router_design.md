# REASON-2B Reasoning Policy Router Design

## Token budget recalculation by stage/section/subsection

```json
{
  "global_stage_budget": {
    "target_scope": "REASON-2B policy router, route strategy, confidence/disagreement scoring, ReasoningController integration patch, tests, docs, tracker, manifest, package, and print sequence.",
    "minimum_complete_version_tokens_est": 15000,
    "deep_implementation_version_tokens_est": 44000,
    "generated_content_total_tokens_est": 98000,
    "max_safe_response_budget_tokens_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-2B-PRINT-P1/P1B/P2/P3.",
    "file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 2,
      "new_tests": 7,
      "new_docs": 6,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "benchmark_count": 0,
    "selected_split_scope": "Full REASON-2B implementation and package; print source files first.",
    "clean_split_points": [
      "source-integrity audit",
      "token-budget subsection declaration",
      "confidence/disagreement scoring",
      "depth route strategy",
      "policy router",
      "ReasoningController integration patch",
      "__init__ export patch",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "evidence/counterfactual reasoning pass integration (REASON-2C)",
      "permanent memory-store commit execution",
      "production external adapter activation",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "unbounded planning/search",
      "fake production-complete claim"
    ]
  },
  "section_budgets": [
    {
      "section": "0 source-of-truth and source-integrity check",
      "target_tokens_est": 1100,
      "status": "complete"
    },
    {
      "section": "1 token budget per section/subsection",
      "target_tokens_est": 1200,
      "status": "complete"
    },
    {
      "section": "2 confidence_disagreement_scoring.py",
      "target_tokens_est": 6200,
      "status": "complete"
    },
    {
      "section": "3 depth_route_strategy.py",
      "target_tokens_est": 7000,
      "status": "complete"
    },
    {
      "section": "4 reasoning_policy_router.py",
      "target_tokens_est": 7600,
      "status": "complete"
    },
    {
      "section": "5 ReasoningController integration patch",
      "target_tokens_est": 5200,
      "status": "complete"
    },
    {
      "section": "6 tests",
      "target_tokens_est": 10500,
      "status": "complete"
    },
    {
      "section": "7 docs/tracker/manifest",
      "target_tokens_est": 9000,
      "status": "complete"
    },
    {
      "section": "8 package and print split",
      "target_tokens_est": 2500,
      "status": "complete"
    }
  ],
  "subsection_budgets": {
    "confidence_disagreement_scoring.py": [
      {
        "subsection": "schemas/enums/config",
        "target_tokens_est": 1400
      },
      {
        "subsection": "finite/bounded checks",
        "target_tokens_est": 1300
      },
      {
        "subsection": "scoring algorithm",
        "target_tokens_est": 1900
      },
      {
        "subsection": "serialization/contracts",
        "target_tokens_est": 1600
      }
    ],
    "depth_route_strategy.py": [
      {
        "subsection": "task/depth route enums",
        "target_tokens_est": 1200
      },
      {
        "subsection": "route plan schema",
        "target_tokens_est": 1500
      },
      {
        "subsection": "deterministic selector",
        "target_tokens_est": 2700
      },
      {
        "subsection": "serialization/contracts",
        "target_tokens_est": 1600
      }
    ],
    "reasoning_policy_router.py": [
      {
        "subsection": "router config and decision schema",
        "target_tokens_est": 1800
      },
      {
        "subsection": "query/context validation",
        "target_tokens_est": 1600
      },
      {
        "subsection": "route + score orchestration",
        "target_tokens_est": 2600
      },
      {
        "subsection": "trace/PAAMA-X metadata/contracts",
        "target_tokens_est": 1600
      }
    ],
    "controller_patch": [
      {
        "subsection": "config extension",
        "target_tokens_est": 900
      },
      {
        "subsection": "optional router init",
        "target_tokens_est": 700
      },
      {
        "subsection": "policy event insertion",
        "target_tokens_est": 1400
      },
      {
        "subsection": "score-informed gate inputs",
        "target_tokens_est": 1200
      },
      {
        "subsection": "compatibility preservation",
        "target_tokens_est": 1000
      }
    ]
  }
}
```

## Source integrity

```json
{
  "primary_source_pack": "/mnt/data/mnemonic_reasoning_reason2a_controller_pack.zip",
  "primary_source_sha256": "86dd0130d1a8e17b9d28725eacdc95ea2f3dd8b356dd452823f2b51e65ec28b3",
  "latest_pack_exists": true,
  "latest_pack_was_not_used_as_primary_reason": "latest development pack did not expose REASON-2A reasoning_controller.py at expected path during source audit",
  "reasoning_controller_exists": true,
  "wm_mann_ltm_adapters_present": true
}
```

## Design

REASON-2B adds an optional, disabled-by-default policy lane to the REASON-2A controller. The new lane has three pieces:

1. `confidence_disagreement_scoring.py` computes bounded confidence/disagreement from finite support scores.
2. `depth_route_strategy.py` selects bounded depth routes by task mode, uncertainty, and conflict flag.
3. `reasoning_policy_router.py` combines scoring and route selection into a JSON-safe policy decision.

The controller patch only activates the router when `ReasoningControllerConfig.use_policy_router=True` and a router config is supplied or enabled. Default REASON-2A behavior remains unchanged. Canonical slot IDs intentionally retain the `reason2a.` prefix for backward compatibility.
