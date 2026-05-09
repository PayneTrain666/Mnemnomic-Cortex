# WM-4A Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated generated artifact text:
- wm_external_memory_interfaces.py: ~6,000-8,000 tokens
- three cross-attention modules: ~12,000-16,000 tokens
- wm_dual_fusion.py: ~5,000-6,500 tokens
- qdt_working_memory.py integration patch: ~800-1,200 tokens
- tests: ~4,000-5,500 tokens
- docs/tracker/continuation: ~3,000-4,000 tokens

Total generated artifact text:
~30,800-41,200 tokens

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
  "wm_external_memory_interfaces.py": false,
  "wm_ltm_cross_attention.py": false,
  "wm_mann_cross_attention.py": false,
  "wm_spcp_cross_attention.py": false,
  "wm_dual_fusion.py": false
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 82%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                          [100%][0m
[32m[32m[1m87 passed[0m[32m in 1.61s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Created wm_external_memory_interfaces.py.
- Created wm_ltm_cross_attention.py.
- Created wm_mann_cross_attention.py.
- Created wm_spcp_cross_attention.py.
- Created wm_dual_fusion.py.
- Patched qdt_working_memory.py to include dual fusion.
- Patched __init__.py exports.
- Added WM-4A tests.
- Updated tracker and deferred register.

Deferred:
- Shared canonical slot store to WM-4B.
- Quantum-holographic storage interface to WM-4C.
- Real external adapters to integration stage.
- Systemwide simultaneous read/write commit gates to WM-5A.

REDO required:
- No

## Ship-check
WM-4A status:
- complete

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for WM-4A cross-memory dual fusion stage.

Tests present:
- tests/test_wm4a_external_memory_interfaces.py
- tests/test_wm4a_cross_attention_dual_fusion.py
- tests/test_wm4a_qdt_integration.py

Tracker blockers:
- None for WM-4A

Stage complete:
- Yes
