# WM-2B Acceptance, Patch Phase, and Ship-Check

## Token budget figures

Estimated generated artifact text:
- wm_intra_depth_transformer.py: ~4,200-5,500 tokens
- wm_cross_depth_transformer.py: ~4,000-5,200 tokens
- depth_specific_addressing.py: ~6,000-7,500 tokens
- tests: ~3,500-4,800 tokens
- docs/tracker/continuation: ~2,500-3,500 tokens

Total generated artifact text:
~20,200-26,500 tokens

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
  "wm_intra_depth_transformer.py": false,
  "wm_cross_depth_transformer.py": false,
  "wm_depth_adapters.py": false,
  "wm_depth_fusion.py": false,
  "wm_trace.py": false,
  "wm_triplet_state.py": false,
  "qdt_working_memory.py": false
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
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[33m                     [100%][0m
[33m=============================== warnings summary ===============================[0m
tests/test_wm2b_quaternion_depth_transformer.py::test_intra_depth_transformer_preserves_shape_and_trace
tests/test_wm2b_quaternion_depth_transformer.py::test_quaternion_replicator_intra_cross_pipeline
tests/test_wm2b_quaternion_depth_transformer.py::test_transformers_reject_bad_shape
  /mnt/data/qdt_wm_maae_wm2b/mnemonic_cortex/working_memory/wm_intra_depth_transformer.py:82: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

tests/test_wm2b_quaternion_depth_transformer.py::test_cross_depth_transformer_preserves_shape_and_trace
tests/test_wm2b_quaternion_depth_transformer.py::test_quaternion_replicator_intra_cross_pipeline
tests/test_wm2b_quaternion_depth_transformer.py::test_transformers_reject_bad_shape
  /mnt/data/qdt_wm_maae_wm2b/mnemonic_cortex/working_memory/wm_cross_depth_transformer.py:83: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
    self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[33m[32m52 passed[0m, [33m[1m6 warnings[0m[33m in 40.90s[0m[0m

```

## DEV-FLOW PATCH PHASE summary

Patched now:
- Created WMIntraDepthTransformer.
- Created WMCrossDepthTransformer.
- Created DepthSpecificAddressing.
- Patched __init__.py exports.
- Added WM-2B tests.
- Updated tracker and deferred register.

Deferred:
- wm_depth_adapters.py, wm_depth_fusion.py, wm_trace.py, wm_triplet_state.py, qdt_working_memory.py to WM-2C.
- Full memory-augmented attention to WM-3A/WM-3B.
- Dual-quaternion SE(3) transport to later spatial/topology stage.

REDO required:
- No

## Ship-check
WM-2B status:
- complete

## Full-Depth Adequacy Gate
Selected scope depth:
- Adequate for WM-2B transformer/addressing stage.

Tests present:
- tests/test_wm2b_quaternion_depth_transformer.py
- tests/test_wm2b_depth_specific_addressing.py

Tracker blockers:
- None for WM-2B

Stage complete:
- Yes
