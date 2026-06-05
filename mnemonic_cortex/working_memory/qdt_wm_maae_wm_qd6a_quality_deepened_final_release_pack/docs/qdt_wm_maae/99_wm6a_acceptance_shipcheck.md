# WM-6A Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated generated artifact text:
- wm_compatibility_wrapper.py: ~4,500-6,000 tokens
- wm_cortex_integration.py: ~5,500-7,500 tokens
- __init__.py export patch: ~500-1,000 tokens
- tests: ~3,500-4,800 tokens
- docs/tracker/continuation: ~3,000-4,000 tokens

Total generated artifact text:
~17,000-23,300 tokens

Practical response print budget:
~7,000-9,000 tokens

Selected strategy:
- Implement full stage in files.
- Run full tests.
- Print summary and next command.
- Provide ZIP with full file contents.

## Source-integrity audit result

```json
{
  "wm_cortex_integration.py": false,
  "wm_compatibility_wrapper.py": false,
  "enhanced_mnemonic_cortex_source_hits": [],
  "real_cortex_source_available": false
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 61%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                           [100%][0m
[32m[32m[1m118 passed[0m[32m in 6.02s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Created wm_compatibility_wrapper.py.
- Created wm_cortex_integration.py.
- Patched __init__.py exports.
- Added WM-6A tests.
- Updated tracker and deferred register.

Partially complete:
- Real EnhancedMnemonicCortex source patch is not applied because no full real source file exists in this pack.
- A tested migration function and patch template are provided.

Deferred:
- Apply migration template to real EnhancedMnemonicCortex source when supplied.
- Persistent storage/transaction logs.
- Real external LTM/MANN/SPCP adapters.
- Final benchmark/release pack to WM-7A.

REDO required:
- No

## Ship-check
WM-6A status:
- complete for available source pack

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for WM-6A integration wrapper/migration stage.

Tests present:
- tests/test_wm6a_compatibility_wrapper.py
- tests/test_wm6a_cortex_integration.py

Tracker blockers:
- None blocking for available source pack; real source patch remains deferred until source is supplied.

Stage complete:
- Yes for available source pack
