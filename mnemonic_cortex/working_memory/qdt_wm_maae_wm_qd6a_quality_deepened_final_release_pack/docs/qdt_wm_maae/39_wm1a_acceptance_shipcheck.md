# WM-1A Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated request budget:
- target output if fully printed in one answer: ~15,000-20,000 tokens
- available practical answer budget: ~8,000-12,000 tokens
- selected execution: create full files now, print core files and patches in response, provide pack for full contents

Implementation token estimate:
- source modules: ~6,500 tokens
- tests: ~2,000 tokens
- docs/commands: ~2,000 tokens
- total generated artifact text: ~10,500 tokens

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

==================================== ERRORS ====================================
[31m[1m___________ ERROR collecting tests/test_wm1a_curved_preservation.py ____________[0m
[31mImportError while importing test module '/mnt/data/qdt_wm_maae_wm1a/tests/test_wm1a_curved_preservation.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
[1m[31m/usr/lib/python3.13/importlib/__init__.py[0m:88: in import_module
    [0m[94mreturn[39;49;00m _bootstrap._gcd_import(name[level:], package, level)[90m[39;49;00m
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[90m[39;49;00m
[1m[31mtests/test_wm1a_curved_preservation.py[0m:3: in <module>
    [0m[94mfrom[39;49;00m[90m [39;49;00m[04m[96mmnemonic_cortex[39;49;00m[04m[96m.[39;49;00m[04m[96mworking_memory[39;49;00m[90m [39;49;00m[94mimport[39;49;00m WMCurvedAssociativeCore, EnhancedCurvedMemory[90m[39;49;00m
[1m[31mE   ImportError: cannot import name 'WMCurvedAssociativeCore' from 'mnemonic_cortex.working_memory' (/mnt/data/qdt_wm_maae_wm1a/mnemonic_cortex/working_memory/__init__.py)[0m[0m
[36m[1m=========================== short test summary info ============================[0m
[31mERROR[0m tests/test_wm1a_curved_preservation.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
[31m[31m[1m1 error[0m[31m in 2.50s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Added canonical-compatible EnhancedCurvedMemory.
- Patched WMCurvedAssociativeCore to delegate to supplied canonical module or internal compatible module.
- Added preservation tests.
- Added export for EnhancedCurvedMemory.

Deferred:
- Curved Resonant WM Core to WM-1B.
- CurvedSlotState and CurvatureMetricPolicy to WM-1C.
- Geometry-aware addressing and bounded spread to WM-1D.
- Local curved trace and shadow writes to WM-1E.

REDO required:
- Yes

## Ship-check

WM-1A status:
- incomplete

## Full-Depth Adequacy Gate

Selected scope depth:
- Adequate for canonical wrapper/preservation stage.

Tests present:
- tests/test_wm1a_curved_preservation.py

Tracker blockers:
- Test failure remains

Stage complete:
- No


## WM-1A patch rerun result

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                            [100%][0m
[32m[32m[1m13 passed[0m[32m in 15.50s[0m[0m

```

WM-1A final patched test status: passed.
