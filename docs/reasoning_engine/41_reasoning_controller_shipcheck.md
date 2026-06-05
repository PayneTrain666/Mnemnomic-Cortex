# REASON-2A Ship-Check and Full-Depth Adequacy Gate

## Ship-check JSON

```json
{
  "stage": "REASON-2A",
  "stage_complete": true,
  "source_pack": "/mnt/data/mnemonic_reasoning_latest_development_integration_pack.zip",
  "source_info": {
    "active_source_pack": "/mnt/data/mnemonic_reasoning_latest_development_integration_pack.zip",
    "active_source_sha256": "415611feeb252100f49141614e6e627567ee228b27c1134591b00ce78e2a319f",
    "repair_log": [
      {
        "file": "mnemonic_cortex/reasoning_depth/shared_depth_slot_registry.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/mann_slotkv_depth_bank.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1c_mann_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/mann_depth_adapter.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1c_mann_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/ltm_depth_banks.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/ltm_depth_adapter.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_capacity_validation.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_integration_readiness.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_lattice_benchmarks.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/working_memory/mann_depth_integration.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "missing_file_copy"
      },
      {
        "file": "mnemonic_cortex/working_memory/ltm_depth_integration.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "missing_file_copy"
      }
    ],
    "wm_depth_controller_exists": true,
    "mann_depth_adapter_exists": true,
    "ltm_depth_adapter_exists": true,
    "shared_depth_slot_registry_exists": true,
    "capacity_validation_exists": true,
    "readiness_exists": true
  },
  "source_files_created": [
    "mnemonic_cortex/reasoning_depth/reasoning_orchestration_trace.py",
    "mnemonic_cortex/reasoning_depth/consolidation_gate.py",
    "mnemonic_cortex/reasoning_depth/reasoning_controller.py"
  ],
  "source_files_patched": [
    "mnemonic_cortex/reasoning_depth/__init__.py",
    "mnemonic_cortex/reasoning_depth/shared_depth_slot_registry.py"
  ],
  "tests_created": [
    "tests/test_reason2a_controller_disabled_default.py",
    "tests/test_reason2a_read_orchestration.py",
    "tests/test_reason2a_mann_hop_routing.py",
    "tests/test_reason2a_ltm_proposal_routing.py",
    "tests/test_reason2a_consolidation_gate.py",
    "tests/test_reason2a_trace_serialization_no_mutation.py",
    "tests/test_reason2a_reason1e_compatibility.py"
  ],
  "docs_created": [
    "docs/reasoning_engine/37_reasoning_controller_design.md",
    "docs/reasoning_engine/38_reasoning_controller_api.md",
    "docs/reasoning_engine/39_consolidation_gate_design.md",
    "docs/reasoning_engine/40_reasoning_controller_tests.md",
    "docs/reasoning_engine/41_reasoning_controller_shipcheck.md",
    "docs/reasoning_engine/42_exact_reason_2b_command.md"
  ],
  "full_test_result": "\u001b[32m\u001b[32m\u001b[1m30 passed\u001b[0m\u001b[32m in 0.97s\u001b[0m\u001b[0m",
  "default_inert_behavior_preserved": true,
  "consolidation_gate_default": "shadow_only",
  "permanent_memory_store_mutation": false,
  "destructive_replacement": false,
  "printout_status": "split_required; PRINT-P1 source files begins in assistant final response",
  "redo_required": false,
  "full_depth_adequacy_gate": "PASS"
}
```

## Pytest output

```text
Spreadsheet runtime warmup failed during python startup
Traceback (most recent call last):
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/patches/warm_spreadsheet_runtime_on_startup.py", line 26, in warm_spreadsheet_runtime_on_startup
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 785, in warm_spreadsheet_runtime
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 720, in _warm_feature_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 704, in _warm_collaboration_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/generated/interface/models.py", line 48821, in hydrate_crdt_from_proto
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/remote.py", line 747, in __call__
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/client.py", line 150, in call
artifact_tool.rpc.client.RemoteError: hydrateCrdtFromProto requires an empty collaborative document.
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                           [100%][0m
[32m[32m[1m30 passed[0m[32m in 0.97s[0m[0m

```

## Patch phase summary

- REASON-2A-PATCH-0001 repaired active source-chain drift by merging missing/stale historical depth files from prior packs.
- REASON-2A-PATCH-0002 restored missing REASON-1E contract exports in package init.
- REASON-2A-PATCH-0003 forced the authoritative REASON-1D registry API required by REASON-1E readiness.
- No remaining P0/P1 blockers discovered inside REASON-2A scope.
- Reasoning controller, orchestration trace, and shadow consolidation gate were implemented.
- Advanced policy/routing remains deferred to REASON-2B and is tracked.

## Audit pack

- No model weights mutated.
- No optimizer state mutated.
- No permanent memory-store mutation performed.
- No destructive WM/MANN/LTM replacement performed.
- No hidden activation introduced; controller defaults disabled.
- Consolidation is shadow/proposal-only by default.
- No fake production-complete claim introduced.
- No fake quantum hardware claim introduced.

## Full-Depth Adequacy Gate

PASS — REASON-2A is deep enough for the first controller/orchestration layer.

## REASON-2B continuation

```text
DEV-FLOW RUN MNEMONIC-REASONING Stage REASON-2B — Reasoning Policy Router, Depth-Route Strategy, Confidence/Disagreement Scoring, and Full File Printout

DEV-FLOW R7N GLOBAL STANDARDS:
- Deep implementation is mandatory by default.
- Recalculate token budget first with actual token figures.
- Token budget is binding, not decorative.
- Create files first, run tests, package ZIP, then print exact generated contents to screen.
- Split at clean source/test/doc boundaries if needed.
- Apply patch phase, audit pack, ship-check, and Full-Depth Adequacy Gate.

SOURCE OF TRUTH:
- Latest development integration pack:
  /mnt/data/mnemonic_reasoning_latest_development_integration_pack.zip
- REASON-2A pack:
  /mnt/data/mnemonic_reasoning_reason2a_controller_pack.zip

PURPOSE:
Extend the REASON-2A controller with explicit reasoning policy routing, depth-route strategy selection, confidence/disagreement scoring, and safer consolidation readiness decisions.

SAFETY:
- Policy/router implementation allowed.
- Confidence/disagreement scoring allowed.
- No permanent memory-store mutation by default.
- No model weight or optimizer mutation.
- No destructive replacement of WM/MANN/LTM.
- No fake production-complete claim.

REQUIRED:
1. Read REASON-2A pack and latest development integration pack.
2. Create reasoning_policy_router.py.
3. Create depth_route_strategy.py.
4. Create confidence_disagreement_scoring.py.
5. Patch ReasoningController to optionally use policy router when enabled.
6. Add tests for default inert policy behavior, route selection, confidence/disagreement scoring, no mutation, trace serialization, and REASON-2A compatibility.
7. Create docs, tracker updates, ship-check, package ZIP, full file printout.
8. Provide exact REASON-2C command for evidence/counterfactual reasoning pass integration.

```
