# Exact WM-2C Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-2C — Depth Fusion, Adapter Restoration, Triplet State, WM Trace, and QDTWorkingMemory Assembly

DEV-FLOW STANDARDS:
- Deep implementation is mandatory by default.
- Recalculate token budget first with token figures.
- Token budget recalculation is binding.
- If deep implementation cannot fit, split at clean source-file boundaries.
- Do not compress source files, tests, docs, or trackers.
- Apply DEV-FLOW PATCH PHASE before ship-check.
- Apply Full-Depth Adequacy Gate before finishing.

TOKEN BUDGET ENFORCEMENT:
- Declare target scope, minimum complete version, deep implementation version, estimated file/module count, expected doc count, expected test count, clean split points, selected split scope, and explicit out-of-scope items.
- If needed, split:
  - WM-2C.1 — WMTrace + WMTripletState + adapters
  - WM-2C.2 — depth fusion + QDTWorkingMemory assembly
  - WM-2C.3 — integration tests + docs/tracker/ship-check

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, true quaternion rotations, context geometry maps, curved core, slot state, metric policy, geometry addressing, bounded spread, local trace, shadow writes, shared-slot doctrine, and QH code requirements.

PAAMA-X REQUIREMENTS:
- Include trace governance, confidence/disagreement hooks, conflict/quarantine hooks, and audit metadata.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

Goal:
Restore and implement assembly-layer modules required to create a usable QDTWorkingMemory.

Required:
1. Create/patch wm_trace.py.
2. Create/patch wm_triplet_state.py.
3. Create/patch wm_depth_adapters.py.
4. Create/patch wm_depth_fusion.py.
5. Create/patch qdt_working_memory.py.
6. Wire QDTWorkingMemory through:
   - WMCurvedAssociativeCore / CurvedResonantWMCore
   - QuaternionDepthReplicator
   - WMIntraDepthTransformer
   - WMCrossDepthTransformer
   - DepthSpecificAddressing
   - CurvedShadowWriteBuffer where available
7. Add tests:
   - shape-safe QDTWorkingMemory read/process/write
   - trace emission
   - depth fusion shape
   - triplet state shape
   - no NaN/Inf
8. Update tracker/deferred work.

Output:
- Token budget recalculation with token figures.
- Source files.
- Tests.
- Patches.
- Tracker updates.
- Deferred work updates.
- Patch phase summary.
- Ship-check.
- Full-Depth Adequacy Gate.
- Exact WM-3A continuation command.
```
