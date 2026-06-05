# Exact WM-QD-2A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-2A — Quaternion Depth, Depth Transformer, Triplet State, Trace, Depth Fusion, and QDTWorkingMemory Assembly Quality Deepening

Goal:
Use the WM-QD-0A/WM-QD-1A quality tooling to harden the quaternion/depth/assembly layer.

Required:
1. Read the latest WM-QD-1A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_quaternion_depth.py
   - wm_intra_depth_transformer.py
   - wm_cross_depth_transformer.py
   - depth_specific_addressing.py
   - wm_depth_adapters.py
   - wm_depth_fusion.py
   - wm_triplet_state.py
   - wm_trace.py
   - qdt_working_memory.py
3. Strengthen shape checks, finite checks, depth/triplet invariants, quaternion normalization guarantees, trace serialization, fallback behavior, and boundedness.
4. Preserve 8-depth default, triplet representation, true quaternion depth rotation, context-map mounting, curved core, and PAAMA-X metadata.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-3A continuation command.

```
