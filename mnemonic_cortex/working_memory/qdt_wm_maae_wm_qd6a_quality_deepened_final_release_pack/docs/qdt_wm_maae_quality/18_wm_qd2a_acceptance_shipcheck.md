# WM-QD-2A Acceptance, Audit, and Ship-Check

## Ship-check JSON

```json
{
  "stage": "WM-QD-2A",
  "stage_complete": false,
  "wm_qd_campaign_complete": false,
  "source_files_produced_or_patched": [
    "wm_depth_guards.py",
    "wm_quaternion_depth.py",
    "wm_intra_depth_transformer.py",
    "wm_cross_depth_transformer.py",
    "depth_specific_addressing.py",
    "wm_depth_adapters.py",
    "wm_depth_fusion.py",
    "wm_triplet_state.py",
    "wm_trace.py",
    "qdt_working_memory.py"
  ],
  "tests_produced": [
    "tests/test_wm_qd2a_depth_guards.py",
    "tests/test_wm_qd2a_module_contracts.py",
    "tests/test_wm_qd2a_qdt_depth_runtime_regression.py",
    "tests/test_wm_qd2a_quality_classifier_scope.py"
  ],
  "docs_produced": [
    "docs/qdt_wm_maae_quality/14_wm_qd2a_depth_assembly_quality_deepening.md",
    "docs/qdt_wm_maae_quality/15_wm_qd2a_deferred_register.md",
    "docs/qdt_wm_maae_quality/16_exact_wm_qd3a_continuation_command.md",
    "docs/qdt_wm_maae_quality/17_wm_qd2a_pytest_output.txt",
    "docs/qdt_wm_maae_quality/18_wm_qd2a_acceptance_shipcheck.md"
  ],
  "before_hashes": {
    "wm_quaternion_depth.py": "8f51fd9b84b485fe68c3e6bbd61cf914eee84e0663b1e7747b00c36cc70ac6d4",
    "wm_intra_depth_transformer.py": "ec393e1c1e6fbc9397806042059fda653ee373f98ddcb5eb0113a53f661bf45b",
    "wm_cross_depth_transformer.py": "8a50687176ae7f8a9b2ea7048dabae7a78ecb94a4b75669aaac5ee453059a4ea",
    "depth_specific_addressing.py": "0dd97fa82ea36662c961f4e701dc30b4e7317d33722616e398ae721e3930e8cd",
    "wm_depth_adapters.py": "27569d8253c346ff4aed816b701f9ace07a9b7166c1af62575c2939c9b10d92a",
    "wm_depth_fusion.py": "cd07cbc2fd45d91b37539ea2831c076e4b7ff03d9e5732972d66c4106e2ff612",
    "wm_triplet_state.py": "504ea0c7b1749be3827dc7a372cb76454c759f3cd853c464bc7d19a5e4b7a1e3",
    "wm_trace.py": "b70ccb1e22f7507912c0710853cb4455990b6a7ff1c2631bc7cfda1a6aecfcc1",
    "qdt_working_memory.py": "7727e366e10bf434fa4dfdbc7c87b540dffb3afe49d21513e390af59bc69d00c"
  },
  "after_hashes": {
    "wm_quaternion_depth.py": "c6e65178a5364f49ed00d69afd20c7234c462f98a1a65cab626098ba1722d5d0",
    "wm_intra_depth_transformer.py": "e623d0d96493e4177e5c2a759a167f6530bd483d94e2fcc6a2d1f63e294f38e1",
    "wm_cross_depth_transformer.py": "68da3a6b5de6f005f8b777edb854e18271120fd9e6c75e6c3099e51cc2e8fe14",
    "depth_specific_addressing.py": "f690a51bab122213b00e635d67141b55f391dd1cbad66d19389ce1fb03353d15",
    "wm_depth_adapters.py": "b0428fc0bfcbb0e1569bf92559bb7f45e5caf3510989575d1c57eba2256b1cf1",
    "wm_depth_fusion.py": "d564a7f9e73f1f3757ba1aca2824d8fd23f4a4bb6e495d40a7b9b42a60eedf21",
    "wm_triplet_state.py": "e20d47eeecef66dbd45a03056c2b9251d3168bc8f84bf27047b4831029f77727",
    "wm_trace.py": "d1eae60f534849feeaa6517da4abdc3f175d60efcf73ea3327aa83b9aa6b192c",
    "qdt_working_memory.py": "c7da07e52a59f4f989a6f34b7dad113e009f4a5e1e4471c878d1250cd40ce615"
  },
  "missing_scope_files": [],
  "redo_required": true
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 48%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31mF[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31m [ 97%]
[0m[32m.[0m[32m.[0m[32m.[0m[31m                                                                      [100%][0m
=================================== FAILURES ===================================
[31m[1m______________________ test_depth_state_rejects_nonfinite ______________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_depth_state_rejects_nonfinite[39;49;00m():[90m[39;49;00m
        depth = torch.randn([94m2[39;49;00m, [94m8[39;49;00m, [94m5[39;49;00m, [94m3[39;49;00m, [94m32[39;49;00m)[90m[39;49;00m
        depth[[94m0[39;49;00m, [94m0[39;49;00m, [94m0[39;49;00m, [94m0[39;49;00m, [94m0[39;49;00m] = [96mfloat[39;49;00m([33m"[39;49;00m[33mnan[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
        [94mwith[39;49;00m pytest.raises(WMDepthValidationError):[90m[39;49;00m
>           ensure_depth_state([33m"[39;49;00m[33mdepth[39;49;00m[33m"[39;49;00m, depth)[90m[39;49;00m

[1m[31mtests/test_wm_qd2a_depth_guards.py[0m:40: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mmnemonic_cortex/working_memory/wm_depth_guards.py[0m:42: in ensure_depth_state
    [0mensure_rank(name, tensor, [94m5[39;49;00m)[90m[39;49;00m
[1m[31mmnemonic_cortex/working_memory/wm_foundation_guards.py[0m:24: in ensure_rank
    [0mensure_finite_tensor(name, tensor)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'depth'
tensor = tensor([[[[[        nan,  1.3581e-02,  1.6598e+00,  ...,  1.1850e+00,
             2.4033e+00,  1.1484e+00],
         ...e+00],
           [-1.9932e-01,  2.0421e+00, -7.9244e-01,  ...,  6.2984e-02,
            -2.7571e-01, -8.0890e-01]]]]])

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mensure_finite_tensor[39;49;00m(name: [96mstr[39;49;00m, tensor: torch.Tensor) -> torch.Tensor:[90m[39;49;00m
    [90m    [39;49;00m[33m"""Validate that a tensor contains no NaN/Inf values."""[39;49;00m[90m[39;49;00m
        [94mif[39;49;00m [95mnot[39;49;00m [96misinstance[39;49;00m(tensor, torch.Tensor):[90m[39;49;00m
            [94mraise[39;49;00m WMFoundationValidationError([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mname[33m}[39;49;00m[33m must be a torch.Tensor[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
        [94mif[39;49;00m [95mnot[39;49;00m torch.isfinite(tensor).all():[90m[39;49;00m
>           [94mraise[39;49;00m WMFoundationValidationError([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mname[33m}[39;49;00m[33m contains NaN or Inf[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
[1m[31mE           mnemonic_cortex.working_memory.wm_foundation_guards.WMFoundationValidationError: depth contains NaN or Inf[0m

[1m[31mmnemonic_cortex/working_memory/wm_foundation_guards.py[0m:19: WMFoundationValidationError
[36m[1m=========================== short test summary info ============================[0m
[31mFAILED[0m tests/test_wm_qd2a_depth_guards.py::[1mtest_depth_state_rejects_nonfinite[0m - mnemonic_cortex.working_memory.wm_foundation_guards.WMFoundationValidationError: depth contains NaN or Inf
[31m[31m[1m1 failed[0m, [32m146 passed[0m[31m in 2.99s[0m[0m

```

## Full-Depth Adequacy Gate

FAIL — REDO/sub-split required due to test failure.

## Next command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-3A — Memory-Augmented Attention, Geometry Scoring, Evidence, Trace, Counterfactual, Conflict, Novelty, and Stability Attention Quality Deepening

Goal:
Use the WM-QD quality tooling to harden the memory-augmented and advanced attention layer.

Required:
1. Read the latest WM-QD-2A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_retrieval_lanes.py
   - wm_geometry_scoring.py
   - wm_memory_augmented_attention.py
   - wm_geometry_linker.py
   - wm_evidence_attention.py
   - wm_trace_attention.py
   - wm_counterfactual_attention.py
   - wm_conflict_attention.py
   - wm_novelty_attention.py
   - wm_stability_attention.py
3. Strengthen candidate schemas, lane output validation, geometry score finite checks, attention boundedness, trace serialization, PAAMA-X metadata, conflict/quarantine hooks, and fallback behavior.
4. Preserve QDTWorkingMemory compatibility and prior depth contracts.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-4A continuation command.

```


## PATCH NOW rerun

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 48%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 97%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m                                                                      [100%][0m
[32m[32m[1m147 passed[0m[32m in 3.40s[0m[0m

```

Final patched status: PASS — WM-QD-2A complete.
