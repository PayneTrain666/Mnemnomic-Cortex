# WM-3A Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated generated artifact text:
- wm_retrieval_lanes.py: ~6,000-7,800 tokens
- wm_geometry_scoring.py: ~3,200-4,200 tokens
- wm_geometry_linker.py: ~1,600-2,200 tokens
- wm_memory_augmented_attention.py: ~4,000-5,200 tokens
- qdt_working_memory.py integration patch: ~500-1,000 tokens
- tests: ~3,500-4,800 tokens
- docs/tracker/continuation: ~3,000-4,000 tokens

Total generated artifact text:
~21,800-29,200 tokens

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
  "wm_retrieval_lanes.py": false,
  "wm_geometry_scoring.py": false,
  "wm_memory_augmented_attention.py": false,
  "wm_geometry_linker.py": false
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31mF[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31m    [100%][0m
=================================== FAILURES ===================================
[31m[1m____ test_qdt_working_memory_read_includes_memory_augmented_attention_trace ____[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_qdt_working_memory_read_includes_memory_augmented_attention_trace[39;49;00m():[90m[39;49;00m
        cfg = QDTWorkingMemoryConfig(input_dim=[94m32[39;49;00m, hidden_dim=[94m64[39;49;00m, num_depths=[94m8[39;49;00m, num_slots=[94m8[39;49;00m, num_heads=[94m4[39;49;00m)[90m[39;49;00m
        wm = QDTWorkingMemory(cfg)[90m[39;49;00m
        x = torch.randn([94m2[39;49;00m, [94m5[39;49;00m, [94m32[39;49;00m)[90m[39;49;00m
    [90m[39;49;00m
        y, trace = wm(x, operation=[33m"[39;49;00m[33mread[39;49;00m[33m"[39;49;00m, return_trace=[94mTrue[39;49;00m)[90m[39;49;00m
    [90m[39;49;00m
        [94massert[39;49;00m y.shape == x.shape[90m[39;49;00m
        [94massert[39;49;00m torch.isfinite(y).all()[90m[39;49;00m
        stages = [item[[33m"[39;49;00m[33mstage[39;49;00m[33m"[39;49;00m] [94mfor[39;49;00m item [95min[39;49;00m trace[[33m"[39;49;00m[33mitems[39;49;00m[33m"[39;49;00m]][90m[39;49;00m
>       [94massert[39;49;00m [33m"[39;49;00m[33mmemory_augmented_attention[39;49;00m[33m"[39;49;00m [95min[39;49;00m stages[90m[39;49;00m
[1m[31mE       AssertionError: assert 'memory_augmented_attention' in ['trace', 'curved_core', 'triplet_state', 'quaternion_depth', 'intra_depth', 'cross_depth', ...][0m

[1m[31mtests/test_wm3a_memory_augmented_attention.py[0m:50: AssertionError
[36m[1m=========================== short test summary info ============================[0m
[31mFAILED[0m tests/test_wm3a_memory_augmented_attention.py::[1mtest_qdt_working_memory_read_includes_memory_augmented_attention_trace[0m - AssertionError: assert 'memory_augmented_attention' in ['trace', 'curved_core', 'triplet_state', 'quaternion_depth', 'intra_depth', 'cross_depth', ...]
[31m[31m[1m1 failed[0m, [32m68 passed[0m[31m in 1.01s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Created wm_retrieval_lanes.py.
- Created wm_geometry_scoring.py.
- Created wm_geometry_linker.py.
- Created wm_memory_augmented_attention.py.
- Patched qdt_working_memory.py to include MAAE on read/process path.
- Patched __init__.py exports.
- Added WM-3A tests.
- Updated tracker and deferred register.

Deferred:
- Evidence/counterfactual/conflict/novelty/stability/trace attention to WM-3B.
- LTM/MANN/SPCP dual fusion to WM-4A.
- Shared slots/QH storage/systemwide commit gates to later stages.

REDO required:
- Yes

## Ship-check
WM-3A status:
- incomplete

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for WM-3A memory-augmented attention foundation.

Tests present:
- tests/test_wm3a_retrieval_lanes_geometry_scoring.py
- tests/test_wm3a_memory_augmented_attention.py

Tracker blockers:
- Test failure remains

Stage complete:
- No


## WM-3A integration patch rerun result

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m    [100%][0m
[32m[32m[1m69 passed[0m[32m in 0.60s[0m[0m

```

WM-3A final patched test status: passed.
