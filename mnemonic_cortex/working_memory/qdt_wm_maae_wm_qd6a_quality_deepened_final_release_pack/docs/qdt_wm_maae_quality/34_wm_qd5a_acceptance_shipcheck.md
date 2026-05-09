# WM-QD-5A Acceptance, Audit, and Ship-Check

## Ship-check JSON

```json
{
  "stage": "WM-QD-5A",
  "stage_complete": false,
  "wm_qd_campaign_complete": false,
  "source_files_produced_or_patched": [
    "wm_commit_cortex_guards.py",
    "wm_system_commit_gate.py",
    "wm_compatibility_wrapper.py",
    "wm_cortex_integration.py",
    "qdt_working_memory.py"
  ],
  "tests_produced": [
    "tests/test_wm_qd5a_commit_cortex_guards.py",
    "tests/test_wm_qd5a_module_contracts.py",
    "tests/test_wm_qd5a_commit_cortex_runtime_regression.py",
    "tests/test_wm_qd5a_quality_classifier_scope.py"
  ],
  "docs_produced": [
    "docs/qdt_wm_maae_quality/30_wm_qd5a_commit_cortex_quality_deepening.md",
    "docs/qdt_wm_maae_quality/31_wm_qd5a_deferred_register.md",
    "docs/qdt_wm_maae_quality/32_exact_wm_qd6a_continuation_command.md",
    "docs/qdt_wm_maae_quality/33_wm_qd5a_pytest_output.txt",
    "docs/qdt_wm_maae_quality/34_wm_qd5a_acceptance_shipcheck.md"
  ],
  "before_hashes": {
    "wm_system_commit_gate.py": "8b142e29c8cb469405cf645169b6395db99d9c707dada9e9bba490446117ec52",
    "wm_compatibility_wrapper.py": "db6b91ea0981bacac9aca13d38a11f9704ff53141be31ac3a5fef7ab47121e37",
    "wm_cortex_integration.py": "9cc461272d8ffc53c7bacf426eb0bbee0aa8c9e355d18238ad0e4ae169ed4c91",
    "qdt_working_memory.py": "c7da07e52a59f4f989a6f34b7dad113e009f4a5e1e4471c878d1250cd40ce615"
  },
  "after_hashes": {
    "wm_system_commit_gate.py": "0caa88ecbfc2e026174a61116179cb39dda0905dbd9a4654e7be5bb5b180ef52",
    "wm_compatibility_wrapper.py": "528c9e84613f6e86ab3fba99255ff33ac05e0f0d3d00f4cca010bda27cd0f47a",
    "wm_cortex_integration.py": "2a0d03ac9c7d67c2aa43c652c941effa5dbf2a10996abc4bc1e56235625f85b2",
    "qdt_working_memory.py": "877e8dec2c6ba158854d420ea416e84b33bed45c8654447ee481f691b47a2ae2"
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 41%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 82%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31mF[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31m                                          [100%][0m
=================================== FAILURES ===================================
[31m[1m_________________ test_commit_proposal_and_decision_validation _________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_commit_proposal_and_decision_validation[39;49;00m():[90m[39;49;00m
        proposal = SystemWriteProposal.create([90m[39;49;00m
            content=torch.randn([94m32[39;49;00m),[90m[39;49;00m
            local_slot_id=[33m"[39;49;00m[33munit[39;49;00m[33m"[39;49;00m,[90m[39;49;00m
            write_permission=[94mTrue[39;49;00m,[90m[39;49;00m
            confidence=[94m0.9[39;49;00m,[90m[39;49;00m
        )[90m[39;49;00m
        ensure_commit_proposal_like([33m"[39;49;00m[33mproposal[39;49;00m[33m"[39;49;00m, proposal, expected_dim=[94m32[39;49;00m)[90m[39;49;00m
    [90m[39;49;00m
        decision = CommitGateDecision([90m[39;49;00m
            proposal_id=proposal.proposal_id,[90m[39;49;00m
            decision=[33m"[39;49;00m[33mcommit[39;49;00m[33m"[39;49;00m,[90m[39;49;00m
            reason=[33m"[39;49;00m[33munit[39;49;00m[33m"[39;49;00m,[90m[39;49;00m
            paamax_metadata={[33m"[39;49;00m[33mdecision[39;49;00m[33m"[39;49;00m: [33m"[39;49;00m[33mcommit[39;49;00m[33m"[39;49;00m},[90m[39;49;00m
        )[90m[39;49;00m
        ensure_commit_decision_like([33m"[39;49;00m[33mdecision[39;49;00m[33m"[39;49;00m, decision)[90m[39;49;00m
    [90m[39;49;00m
        bad = SystemWriteProposal.create(content=torch.randn([94m32[39;49;00m), write_permission=[94mTrue[39;49;00m, confidence=[94m0.9[39;49;00m)[90m[39;49;00m
        bad.content[[94m0[39;49;00m] = [96mfloat[39;49;00m([33m"[39;49;00m[33mnan[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
        [94mwith[39;49;00m pytest.raises(WMCommitCortexValidationError):[90m[39;49;00m
>           ensure_commit_proposal_like([33m"[39;49;00m[33mbad[39;49;00m[33m"[39;49;00m, bad, expected_dim=[94m32[39;49;00m)[90m[39;49;00m

[1m[31mtests/test_wm_qd5a_commit_cortex_guards.py[0m:42: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
[1m[31mmnemonic_cortex/working_memory/wm_commit_cortex_guards.py[0m:49: in ensure_commit_proposal_like
    [0mensure_finite_tensor([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mname[33m}[39;49;00m[33m.content[39;49;00m[33m"[39;49;00m, content)[90m[39;49;00m
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'bad.content'
tensor = tensor([    nan, -0.6918, -1.0188,  1.4737,  0.2352,  0.5625,  0.0847, -0.6125,
         0.6292,  0.1285, -1.3037, -1....6, -0.5179,  0.7901,  0.2978, -0.1732,
        -0.3014,  0.1962,  0.5173, -0.3514,  0.1089, -0.3599, -0.2914, -1.7650])

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mensure_finite_tensor[39;49;00m(name: [96mstr[39;49;00m, tensor: torch.Tensor) -> torch.Tensor:[90m[39;49;00m
    [90m    [39;49;00m[33m"""Validate that a tensor contains no NaN/Inf values."""[39;49;00m[90m[39;49;00m
        [94mif[39;49;00m [95mnot[39;49;00m [96misinstance[39;49;00m(tensor, torch.Tensor):[90m[39;49;00m
            [94mraise[39;49;00m WMFoundationValidationError([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mname[33m}[39;49;00m[33m must be a torch.Tensor[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
        [94mif[39;49;00m [95mnot[39;49;00m torch.isfinite(tensor).all():[90m[39;49;00m
>           [94mraise[39;49;00m WMFoundationValidationError([33mf[39;49;00m[33m"[39;49;00m[33m{[39;49;00mname[33m}[39;49;00m[33m contains NaN or Inf[39;49;00m[33m"[39;49;00m)[90m[39;49;00m
[1m[31mE           mnemonic_cortex.working_memory.wm_foundation_guards.WMFoundationValidationError: bad.content contains NaN or Inf[0m

[1m[31mmnemonic_cortex/working_memory/wm_foundation_guards.py[0m:19: WMFoundationValidationError
[36m[1m=========================== short test summary info ============================[0m
[31mFAILED[0m tests/test_wm_qd5a_commit_cortex_guards.py::[1mtest_commit_proposal_and_decision_validation[0m - mnemonic_cortex.working_memory.wm_foundation_guards.WMFoundationValidationError: bad.content contains NaN or Inf
[31m[31m[1m1 failed[0m, [32m174 passed[0m[31m in 5.19s[0m[0m

```

## Full-Depth Adequacy Gate

FAIL — REDO/sub-split required due to test failure.

## Next command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-6A — Final Release, Benchmark, Production Readiness, Deferred Hardening Plan, and Quality Campaign Closure

Goal:
Use the WM-QD quality tooling to finalize the quality-deepened WM release/readiness pack.

Required:
1. Read the latest WM-QD-5A quality pack.
2. Re-run full source/API audit and module dependency audit.
3. Re-run full tests.
4. Re-run or update smoke benchmark harness.
5. Verify all WM-QD contracts:
   - WM-QD-1A foundation contracts
   - WM-QD-2A depth contracts
   - WM-QD-3A attention contracts
   - WM-QD-4A external-memory/shared/QH contracts
   - WM-QD-5A commit/cortex contracts
6. Produce final release manifest.
7. Produce final production readiness document.
8. Produce final deferred hardening plan.
9. Confirm remaining real-source integration requirements honestly.
10. Update quality tracker/deferred register.
11. Package final quality-deepened release ZIP.
12. Produce final ship-check and campaign closure statement or exact REDO command.

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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 41%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 82%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m                                          [100%][0m
[32m[32m[1m175 passed[0m[32m in 5.39s[0m[0m

```

Final patched status: PASS — WM-QD-5A complete.
