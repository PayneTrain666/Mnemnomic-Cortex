# WM-2B Quaternion Depth Transformer and Depth-Specific Addressing

## Source-integrity audit

| module | exists before WM-2B |
|---|---|
| `wm_intra_depth_transformer.py` | False |
| `wm_cross_depth_transformer.py` | False |
| `wm_depth_adapters.py` | False |
| `wm_depth_fusion.py` | False |
| `wm_trace.py` | False |
| `wm_triplet_state.py` | False |
| `qdt_working_memory.py` | False |

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_intra_depth_transformer.py
mnemonic_cortex/working_memory/wm_cross_depth_transformer.py
mnemonic_cortex/working_memory/depth_specific_addressing.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm2b_quaternion_depth_transformer.py
tests/test_wm2b_depth_specific_addressing.py
```

## Implemented behavior

### WMIntraDepthTransformer
- input/output: [B,Z,T,3,D]
- transformer operates within each depth/triplet temporal stream.
- trace includes stream count, temporal delta norm, finite status, and PAAMA-X metadata.

### WMCrossDepthTransformer
- input/output: [B,Z,T,3,D]
- transformer operates across depth slices for each batch/time/triplet group.
- trace includes depth sequence count, depth energy proxy, finite status, and PAAMA-X metadata.

### DepthSpecificAddressing
- input: [B,Z,T,3,D]
- output activation: [B,Z,S]
- uses depth summaries, CurvedSlotStateBank, CurvatureMetricPolicy, and context geometry maps.
- trace includes selected slots per depth, geometry_by_depth, curvature bias shape, and PAAMA-X metadata.

## Explicitly deferred
- wm_depth_adapters.py, wm_depth_fusion.py, wm_trace.py, wm_triplet_state.py, and qdt_working_memory.py are deferred to WM-2C.
- full memory-augmented attention modules remain deferred to WM-3A/WM-3B.
