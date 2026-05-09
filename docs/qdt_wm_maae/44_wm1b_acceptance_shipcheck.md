# WM-1B Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated request budget:
- source implementation: ~4,500-6,000 tokens
- tests: ~1,500-2,000 tokens
- docs/tracker/continuation: ~2,000-2,800 tokens
- total artifact text: ~8,000-10,800 tokens

Practical response print budget:
- ~7,000-9,000 tokens

Selected strategy:
- Create full files and pack.
- Print key file contents and results in response.
- Provide ZIP for all files.

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                       [100%][0m
[32m[32m[1m18 passed[0m[32m in 20.76s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Added CurvedResonantWMCore.
- Added bounded resonance loop.
- Added novelty/lightbulb scoring.
- Added resonance trace events.
- Added PAAMA-X metadata.
- Added preservation of WMCurvedAssociativeCore as inner core.
- Added tests.

Deferred:
- CurvedSlotState + CurvatureMetricPolicy to WM-1C.
- Geometry-aware addressing + bounded spread to WM-1D.
- Curved shadow writes to WM-1E.

REDO required:
- No

## Ship-check
WM-1B status:
- complete

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for Curved Resonant WM Core wrapper stage.

Tests present:
- tests/test_wm1b_curved_resonant_core.py

Tracker blockers:
- None

Stage complete:
- Yes
