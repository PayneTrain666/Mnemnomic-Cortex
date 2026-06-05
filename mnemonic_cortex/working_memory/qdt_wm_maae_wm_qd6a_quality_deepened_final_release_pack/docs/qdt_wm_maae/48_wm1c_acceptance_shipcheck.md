# WM-1C Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated generated artifact text:
- curved_slot_state.py: ~5,500-7,000 tokens
- curvature_metric_policy.py: ~3,500-4,800 tokens
- tests/test_wm1c_slot_state_policy.py: ~1,800-2,400 tokens
- docs/tracker/continuation: ~2,000-2,800 tokens

Total generated artifact text:
~12,800-17,000 tokens

Practical response print budget:
~7,000-9,000 tokens

Selected strategy:
- Create all files.
- Run full tests.
- Print summary and continuation command in response.
- Provide ZIP with all files.

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                  [100%][0m
[32m[32m[1m23 passed[0m[32m in 23.08s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Added CurvedSlotStateBank.
- Added CurvedSlotSnapshot and CurvedSlotStateTrace.
- Added stable tensor views, validation, and repair.
- Added CurvatureMetricPolicy.
- Added global/per-slot/per-depth/context curvature.
- Added clamps and drift penalty.
- Added PAAMA-X metadata in curvature trace.
- Added tests.

Deferred:
- Direct geometry-aware addressing integration to WM-1D.
- Curved shadow writes to WM-1E.

REDO required:
- No

## Ship-check
WM-1C status:
- complete

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for CurvedSlotState and CurvatureMetricPolicy stage.

Tests present:
- tests/test_wm1c_slot_state_policy.py

Tracker blockers:
- None

Stage complete:
- Yes
