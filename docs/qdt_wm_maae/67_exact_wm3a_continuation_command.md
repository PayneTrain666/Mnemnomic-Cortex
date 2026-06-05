# Exact WM-3A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-3A — Memory-Augmented Attention Engine, Retrieval Lanes, Geometry Scoring, and PAAMA-X Policy Lane

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
  - WM-3A.1 — retrieval lanes + lane outputs
  - WM-3A.2 — geometry scoring + memory augmented attention
  - WM-3A.3 — PAAMA-X policy lane + tests/docs/ship-check

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, true quaternion rotations, context geometry maps, curved core, QDTWorkingMemory assembly, shared-slot doctrine, and QH code requirements.

PAAMA-X REQUIREMENTS:
- Include policy lane, trace governance, write-permission hooks, confidence/disagreement hooks, conflict/quarantine hooks, and audit metadata.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

Goal:
Implement the first full memory-augmented attention layer for QDTWorkingMemory.

Required:
1. Create/patch wm_retrieval_lanes.py.
2. Create/patch wm_geometry_scoring.py.
3. Create/patch wm_memory_augmented_attention.py.
4. Create/patch wm_geometry_linker.py if needed for lane routing.
5. Implement lanes:
   - vector
   - hyperbolic
   - temporal
   - spatial
   - procedural
   - trace
   - policy
6. Implement geometry-aware scoring over retrieved candidates.
7. Implement PAAMA-X policy lane metadata and write-permission hooks.
8. Add tests for lane shapes, scoring, policy lane, trace output, and QDTWorkingMemory compatibility.
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
- Exact WM-3B continuation command.
```
