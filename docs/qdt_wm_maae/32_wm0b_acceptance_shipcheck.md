# WM-0B Acceptance, Patch Phase, and Ship-Check

## Token budget recalculation

Target scope:
- Implement context geometry maps, selector, triplet projection, context-to-WM bridge, stability guard, trace schema, and tests.

Minimum complete version:
- Full preset family, selector, mounting bridge, stability trace, and tests.

Deep implementation version:
- Implemented as source modules plus tests, while leaving external topology-manager integration deferred.

Estimated file/module count:
- Source modules: 8 created/updated.
- Docs: 4 updated/created.
- Tests: 3 added.

Clean split points:
- WM-0B fits in one implementation pack.

Selected split scope:
- Context-buffer geometry maps and mounting only.

Explicit out-of-scope:
- Canonical curved WM wrapper.
- Curved Resonant WM Core.
- Full topology manager.
- Cortex integration.

## DEV-FLOW PATCH PHASE summary

Patched now:
- Expanded context maps from 3 minimal presets to 10 full presets.
- Added PAAMA-X policy/governance map.
- Added quantum-holographic context map.
- Added context triplet projection.
- Added context depth adapter.
- Added context stability guard.
- Added context mount trace schema.
- Added tests.

Deferred:
- Full topology-manager integration.
- Learned selector training.
- Curved core wrapper.

REDO required:
- No.

## Ship-check
WM-0B complete for selected scope.

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for context-map implementation.

Tests present:
- test_wm0b_context_geometry_maps.py
- test_wm0b_context_selector_bridge.py
- test_wm0b_geometry_mounted_context_buffer.py

Tracker blockers:
- None for WM-0B.

Redo required:
- No.

Stage complete:
- Yes.

Next command:
- WM-1A.


## Patch rerun result

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31mF[0m[32m.[0m[32m.[0m[31m                                                                 [100%][0m
=================================== FAILURES ===================================
[31m[1m_________ test_geometry_mounted_context_buffer_preserves_backward_api __________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_geometry_mounted_context_buffer_preserves_backward_api[39;49;00m():[90m[39;49;00m
        buffer = GeometryMountedContextBuffer(dim=[94m32[39;49;00m, num_depths=[94m8[39;49;00m)[90m[39;49;00m
        ctx = torch.randn([94m2[39;49;00m, [94m4[39;49;00m, [94m32[39;49;00m)[90m[39;49;00m
        depth_state = torch.randn([94m2[39;49;00m, [94m8[39;49;00m, [94m5[39;49;00m, [94m3[39;49;00m, [94m32[39;49;00m)[90m[39;49;00m
>       mounted, selected = buffer.mount(ctx, depth_state, requested_map=[33m"[39;49;00m[33mquantum_holographic[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[90m[39;49;00m

[1m[31mtests/test_wm0b_geometry_mounted_context_buffer.py[0m:10: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mmnemonic_cortex/working_memory/wm_context_mount.py[0m:65: in mount
    [0mselected.trace = trace  [90m# dynamic compatibility field for callers/tests[39;49;00m[90m[39;49;00m
    ^^^^^^^^^^^^^^[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = ContextGeometryMap(name='quantum_holographic', purpose='Depth/bank/geometry/triplet-coded holographic read/write prepa...ht': 1.25, 'min_depth_weight': 0.0, 'requires_trace': 1.0}, mount_strategy='phase_bias', triplet_bias=[1.0, 0.5, 0.25])
name = 'trace'
value = ContextMountTrace(selected_map='quantum_holographic', selection_reason='requested_map', geometry_by_depth=['euclidean'... 0.0, 'conflict_verification': 0.0, 'creative_synthesis': 0.0, 'policy_governance': 0.0, 'quantum_holographic': 999.0})

>   [0m[04m[91m?[39;49;00m[04m[91m?[39;49;00m[04m[91m?[39;49;00m[90m[39;49;00m
[1m[31mE   dataclasses.FrozenInstanceError: cannot assign to field 'trace'[0m

[1m[31m<string>[0m:25: FrozenInstanceError
[36m[1m=========================== short test summary info ============================[0m
[31mFAILED[0m tests/test_wm0b_geometry_mounted_context_buffer.py::[1mtest_geometry_mounted_context_buffer_preserves_backward_api[0m - dataclasses.FrozenInstanceError: cannot assign to field 'trace'
[31m[31m[1m1 failed[0m, [32m7 passed[0m[31m in 13.01s[0m[0m

```

WM-0B patched test status: failed; REDO required.


## Final patched rerun result

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                                 [100%][0m
[32m[32m[1m8 passed[0m[32m in 14.79s[0m[0m

```

WM-0B final patched test status: passed.
