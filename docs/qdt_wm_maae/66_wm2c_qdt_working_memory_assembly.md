# WM-2C Depth Fusion, Adapter Restoration, Triplet State, WM Trace, and QDTWorkingMemory Assembly

## Source-integrity audit before WM-2C

| module | exists before WM-2C |
|---|---|
| `wm_config.py` | False |
| `wm_trace.py` | False |
| `wm_triplet_state.py` | False |
| `wm_depth_adapters.py` | False |
| `wm_depth_fusion.py` | False |
| `qdt_working_memory.py` | False |

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_config.py
mnemonic_cortex/working_memory/wm_trace.py
mnemonic_cortex/working_memory/wm_triplet_state.py
mnemonic_cortex/working_memory/wm_depth_adapters.py
mnemonic_cortex/working_memory/wm_depth_fusion.py
mnemonic_cortex/working_memory/qdt_working_memory.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm2c_trace_triplet_adapters_fusion.py
tests/test_wm2c_qdt_working_memory_assembly.py
```

## Implemented behavior

### WMTrace
- trace items
- PAAMA-X metadata aggregation
- confidence/disagreement scores
- serializable summary

### WMTripletState
- anchor/direction/phase state
- projection/fusion
- shape summaries

### WMDepthAdapters
- per-depth/per-triplet adapter stack for [B,Z,T,3,D]
- trace and finite validation

### WMDepthFusion
- fuses [B,Z,T,3,D] back to [B,T,D]
- trainable depth/triplet weights
- disagreement proxy
- trace metadata

### QDTWorkingMemory
Wires:
- CurvedResonantWMCore
- CurvedShadowWriteBuffer
- QuaternionDepthReplicator
- WMIntraDepthTransformer
- WMCrossDepthTransformer
- WMDepthAdapters
- DepthSpecificAddressing
- WMDepthFusion
- WMTraceEmitter
