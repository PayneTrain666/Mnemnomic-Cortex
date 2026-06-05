# Exact WM-4A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-4A — LTM/MANN/SPCP Cross-Attention Interfaces and Dual Fusion Controller

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
  - WM-4A.1 — external memory interface contracts
  - WM-4A.2 — LTM/MANN/SPCP cross-attention + trace visibility
  - WM-4A.3 — dual fusion controller + tests/docs/ship-check

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, true quaternion rotations, context geometry maps, curved core, QDTWorkingMemory assembly, MAAE lanes, advanced attention, shared-slot doctrine, and QH code requirements.

PAAMA-X REQUIREMENTS:
- Include trace governance, write-permission hooks, confidence/disagreement hooks, conflict/quarantine hooks, audit metadata, and policy-lane integration.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

Goal:
Implement WM cross-attention interfaces for LTM, MANN, and SPCP plus WM dual-fusion controller.

Required:
1. Create/patch wm_external_memory_interfaces.py.
2. Create/patch wm_ltm_cross_attention.py.
3. Create/patch wm_mann_cross_attention.py.
4. Create/patch wm_spcp_cross_attention.py.
5. Create/patch wm_dual_fusion.py.
6. Ensure MANN trace visibility:
   - pre-fusion outputs
   - per-hop attention
   - scratchpad tokens
   - confidence
   - disagreement
7. Integrate dual-fusion output into QDTWorkingMemory trace path.
8. Add tests for interface contracts, fusion shape, trace visibility, confidence/disagreement, and no NaN/Inf.
9. Update tracker/deferred work.

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
- Exact WM-4B continuation command.
```
