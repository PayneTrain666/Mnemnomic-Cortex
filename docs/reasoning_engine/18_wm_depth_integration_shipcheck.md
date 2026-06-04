# REASON-1B Ship-Check and Full-Depth Adequacy Gate

## Ship-check JSON

```json
{
  "stage": "REASON-1B",
  "stage_complete": true,
  "source_pack": "/mnt/data/mnemonic_reasoning_reason1a_depth_lattice_pack.zip",
  "source_info": {
    "reason1a_pack_exists": true,
    "context_pack_exists": true,
    "wm_pack_exists": true,
    "active_source_pack": "/mnt/data/mnemonic_reasoning_reason1a_depth_lattice_pack.zip",
    "reason1a_sha256": "a614a7289913c2636f7e03f3b5bc4ed9b1387d06c0c227dcc1fb553e0cbe99b4",
    "context_pack_sha256": "078ffc7874e4bca1c996e1a3c5569f26e7be712901dbbbbf3def7dfb7393e02f",
    "wm_pack_sha256": "1302773410ceb42172f9fc7b1b26411ff4ed3aad8f126c25d6c6f5de58f90dac",
    "qdt_working_memory_source_exists": true,
    "context_compression_source_exists": true,
    "wm_context_mount_source_exists": true
  },
  "source_files_created": [
    "mnemonic_cortex/reasoning_depth/wm_depth_adapter.py",
    "mnemonic_cortex/reasoning_depth/wm_depth_controller.py",
    "mnemonic_cortex/working_memory/wm_depth_integration.py"
  ],
  "source_files_patched": [
    "mnemonic_cortex/reasoning_depth/__init__.py",
    "mnemonic_cortex/working_memory/__init__.py"
  ],
  "tests_created": [
    "tests/test_reason1b_wm_depth_adapter_disabled.py",
    "tests/test_reason1b_wm_depth_adapter_enabled.py",
    "tests/test_reason1b_context_candidate_routing.py",
    "tests/test_reason1b_wm_depth_controller.py",
    "tests/test_reason1b_wm_depth_integration.py",
    "tests/test_reason1b_qdt_compatibility.py"
  ],
  "docs_created": [
    "docs/reasoning_engine/15_wm_depth_integration_design.md",
    "docs/reasoning_engine/16_wm_depth_integration_api.md",
    "docs/reasoning_engine/17_wm_depth_integration_tests.md",
    "docs/reasoning_engine/18_wm_depth_integration_shipcheck.md",
    "docs/reasoning_engine/19_exact_reason_1c_command.md"
  ],
  "full_test_result": "\u001b[32m\u001b[32m\u001b[1m22 passed\u001b[0m\u001b[32m in 0.61s\u001b[0m\u001b[0m",
  "default_inert_behavior_preserved": true,
  "destructive_qdt_replacement": false,
  "context_candidate_routes": [
    "Z3 contextual_binding",
    "Z5 temporal_episode",
    "Z7 volatile_trace"
  ],
  "default_write_mutation": false,
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                   [100%][0m
[32m[32m[1m22 passed[0m[32m in 0.61s[0m[0m

```

## Full-Depth Adequacy Gate

PASS — REASON-1B is deep enough for optional WM depth integration.

## REASON-1C continuation

```text
DEV-FLOW RUN MNEMONIC-REASONING Stage REASON-1C — Integrate DepthIndexedSlotLattice into MANN SlotKV Reasoning Path

SOURCE OF TRUTH:
- REASON-1B pack:
  /mnt/data/mnemonic_reasoning_reason1b_wm_depth_integration_pack.zip
- Current WM/QDT source baseline:
  /mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip

PURPOSE:
Integrate slots × 8 depth capacity into the MANN reasoning scratchpad as a depth-indexed SlotKV bank while preserving existing MANN behavior by default.

DEV-FLOW STANDARDS:
- Deep implementation mandatory.
- Recalculate token budget first with actual figures.
- Create files, run tests, package ZIP, and print all generated contents.
- Split printout at file boundaries if needed.
- Apply patch phase and Full-Depth Adequacy Gate.

SAFETY:
- No destructive MANN replacement.
- No direct shared physical tensor storage with LTM.
- MANN depth lattice disabled/inert unless enabled by config.
- MANN writes remain shadow/proposal-only unless explicit gate permits mutation.

REQUIRED:
1. Read REASON-1B pack.
2. Create mann_depth_adapter.py.
3. Create mann_slotkv_depth_bank.py if needed.
4. Implement keys [S,8,K], values [S,8,V] MANN read/write proposals.
5. Implement hop-oriented read traces: selected slots, selected depths, depth entropy, support mass, confidence, disagreement.
6. Add tests for disabled default, enabled MANN depth read, SlotKV shape, no mutation default, trace serialization, and shared canonical IDs without shared tensors.
7. Create docs, tracker updates, ship-check, package ZIP, full file printout.
8. Provide exact REASON-1D command for LTM depth adapter integration.

```
