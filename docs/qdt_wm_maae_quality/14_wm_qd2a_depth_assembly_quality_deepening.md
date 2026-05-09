# WM-QD-2A Quaternion Depth / Assembly Quality Deepening

## Scope

- `wm_quaternion_depth.py`
- `wm_intra_depth_transformer.py`
- `wm_cross_depth_transformer.py`
- `depth_specific_addressing.py`
- `wm_depth_adapters.py`
- `wm_depth_fusion.py`
- `wm_triplet_state.py`
- `wm_trace.py`
- `qdt_working_memory.py`

## Patch summary

```json
{
  "stage": "WM-QD-2A",
  "source_pack": "/mnt/data/qdt_wm_maae_wm_qd1a_foundation_quality_pack.zip",
  "scope_files": [
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
  "present_before_patch": {
    "wm_quaternion_depth.py": true,
    "wm_intra_depth_transformer.py": true,
    "wm_cross_depth_transformer.py": true,
    "depth_specific_addressing.py": true,
    "wm_depth_adapters.py": true,
    "wm_depth_fusion.py": true,
    "wm_triplet_state.py": true,
    "wm_trace.py": true,
    "qdt_working_memory.py": true
  },
  "missing_scope_files": [],
  "patched_files": [
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
  "safety": {
    "runtime_modules_patched_in_scope": true,
    "no_model_weight_mutation": true,
    "no_optimizer_mutation": true,
    "no_external_adapter_activation": true,
    "no_fake_production_claim": true
  },
  "token_budget": {
    "target": "Quaternion/depth/assembly hardening over depth guards, contracts, tests, and QDT regression.",
    "minimum_complete": "Depth guards, module contracts, tests, docs, tracker/deferred updates, full test run.",
    "deep_version": "Shared depth guard module plus explicit depth contracts on every in-scope module and runtime regression tests.",
    "max_response_budget": "summary only; source in ZIP",
    "split_decision": "No sub-split required."
  }
}
```

## What was strengthened

- Added `wm_depth_guards.py` with [B,T,D] token validation, [B,Z,T,3,D] depth validation, triplet-axis validation, quaternion normalization checks, summaries, compatibility checks, and depth contract traces.
- Added `wm_qd2a_depth_contract()` to every present depth/assembly module.
- Added PAAMA-X-compatible depth contract metadata.
- Added runtime regression tests for QDTWorkingMemory read/process/write.
- Added bounded classifier/remediation tests for WM-QD-2A scope.
